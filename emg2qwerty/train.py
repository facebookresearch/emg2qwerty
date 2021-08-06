import logging
import os
import pprint
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, List, Tuple

import hydra
import numpy as np
import pytorch_lightning as pl
import torch
from hydra.utils import get_original_cwd, instantiate
from omegaconf import DictConfig, ListConfig, OmegaConf
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader

from emg2qwerty import transforms, utils
from emg2qwerty.charset import charset
from emg2qwerty.data import WindowedEmgDataset
from emg2qwerty.metrics import CharacterErrorRate
from emg2qwerty.modules import (MultiBandRotationInvariantMLP, SpectrogramNorm,
                                TDSConvEncoder)
from emg2qwerty.transforms import Transform

log = logging.getLogger(__name__)


class WindowedEmgDataModule(pl.LightningDataModule):
    def __init__(
        self,
        window_length: int,
        padding: Tuple[int, int],
        batch_size: int,
        num_workers: int,
        train_sessions: Sequence[Path],
        val_sessions: Sequence[Path],
        test_sessions: Sequence[Path],
    ) -> None:
        super().__init__()

        self.window_length = window_length
        self.padding = padding

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_sessions = train_sessions
        self.val_sessions = val_sessions
        self.test_sessions = test_sessions

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_dataset = ConcatDataset([
            WindowedEmgDataset(
                hdf5_path,
                transform=self._train_transforms,
                window_length=self.window_length,
                padding=self.padding,
            ) for hdf5_path in self.train_sessions
        ])
        self.val_dataset = ConcatDataset([
            WindowedEmgDataset(
                hdf5_path,
                transform=self._val_transforms,
                window_length=self.window_length,
                padding=self.padding,
            ) for hdf5_path in self.val_sessions
        ])
        self.test_dataset = ConcatDataset([
            WindowedEmgDataset(hdf5_path, transform=self._test_transforms)
            for hdf5_path in self.test_sessions
        ])

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=WindowedEmgDataset.collate,
            pin_memory=True,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=WindowedEmgDataset.collate,
            pin_memory=True,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader:
        # At test time, feed entire sessions at once without windowing/padding
        # for more realism. Limit batch size to 1 to fit within GPU memory and
        # avoid any influence of padding in test scores.
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            num_workers=self.num_workers,
            collate_fn=WindowedEmgDataset.collate,
            pin_memory=True,
            shuffle=False,
        )


class TDSConvCTCModule(pl.LightningModule):
    def __init__(
        self,
        in_features: int,
        mlp_features: int,
        block_channels: Sequence[int],
        kernel_width: int,
        optimizer: DictConfig,
        lr_scheduler: DictConfig,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        # classes = character set + CTC blank token
        self.num_classes = charset().num_classes + 1
        self.blank_label = charset().null_class

        # Constants for readability
        num_bands = 2
        in_channels = 16  # Electrode channels
        num_features = mlp_features[-1] * num_bands

        # Model
        self.spec_norm = SpectrogramNorm(channels=in_channels * num_bands)
        self.rotation_invariant_mlp = MultiBandRotationInvariantMLP(
            num_bands=num_bands,
            in_features=in_features,
            mlp_features=mlp_features,
            pooling="mean")
        self.conv_encoder = TDSConvEncoder(num_features=num_features,
                                           block_channels=block_channels,
                                           kernel_width=kernel_width)
        self.linear = nn.Linear(num_features, self.num_classes)
        self.log_softmax = nn.LogSoftmax(dim=-1)

        # Criterion
        self.ctc_loss = nn.CTCLoss(blank=self.blank_label)

        # Metric
        self.cer = CharacterErrorRate()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs  # (T, N, bands=2, freq, electrode_channels=16)
        x = self.spec_norm(x)  # (T, N, bands=2, freq, C=16)
        x = self.rotation_invariant_mlp(x)  # (T, N, bands=2, mlp_features[-1])
        x = x.flatten(start_dim=2)  # (T, N, num_features)
        x = self.conv_encoder(x)  # (T, N, num_features)
        x = self.linear(x)  # (T, N, num_classes)
        return self.log_softmax(x)  # (T, N, num_classes)

    def _step(self, phase: str, batch: Mapping[str, torch.Tensor], *args,
              **kwargs) -> torch.Tensor:
        inputs = batch["inputs"]
        targets = batch["targets"]
        input_lengths = batch["input_lengths"]
        target_lengths = batch["target_lengths"]
        N = len(input_lengths)  # batch_size

        emissions = self.forward(inputs)

        # Shrink input lengths by an amount equivalent to the
        # conv encoder's temporal receptive field to compute output
        # activation lengths for CTCLoss
        T_diff = inputs.shape[0] - emissions.shape[0]
        emission_lengths = input_lengths - T_diff

        loss = self.ctc_loss(
            log_probs=emissions,  # (T, N, num_classes)
            targets=targets.transpose(0, 1),  # (T, N) -> (N, T)
            input_lengths=emission_lengths,  # (N,)
            target_lengths=target_lengths,  # (N,)
        )

        # Decode emissions and update CER metric
        for i in range(N):
            # Unpad emission matrix for batch entry and perform CTC decoding.
            # emissions: (T, N, num_classes)
            emission = emissions[:emission_lengths[i], i]
            pred_labels = self._ctc_greedy_decode(emission)

            # Unpad target labels for batch entry. targets: (T, N)
            target_labels = targets[:target_lengths[i], i]

            self.cer.update(pred_labels, target_labels)

        self.log(f"{phase}_loss", loss, sync_dist=True)
        return loss

    def _epoch_end(self, phase: str, outputs: Sequence[Any]) -> None:
        self.log(f"{phase}_CER", self.cer.compute(), sync_dist=True)
        self.cer.reset()

    def training_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("train", *args, **kwargs)

    def validation_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("val", *args, **kwargs)

    def test_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("test", *args, **kwargs)

    def training_epoch_end(self, outputs: Sequence[Any]) -> None:
        self._epoch_end("train", outputs)

    def validation_epoch_end(self, outputs: Sequence[Any]) -> None:
        self._epoch_end("val", outputs)

    def test_epoch_end(self, outputs: Sequence[Any]) -> None:
        self._epoch_end("test", outputs)

    def configure_optimizers(self):
        return utils.instantiate_optimizer_and_scheduler(
            self.parameters(),
            optimizer_config=self.hparams.optimizer,
            lr_scheduler_config=self.hparams.lr_scheduler,
        )

    def _ctc_greedy_decode(self, emission: torch.Tensor) -> List[int]:
        # emission: (T, num_classes)
        assert emission.ndim == 2
        assert emission.shape[1] == self.num_classes

        decoding = []
        prev_label = self.blank_label
        for label in emission.argmax(-1).detach().cpu().numpy():
            if label != self.blank_label and label != prev_label:
                decoding.append(label)
            prev_label = label

        return decoding


@hydra.main(config_path="../config", config_name="base")
def main(config: DictConfig):
    log.info(f"\nConfig:\n{OmegaConf.to_yaml(config)}")

    # Add working dir to PYTHONPATH
    working_dir = get_original_cwd()
    python_paths = os.environ.get("PYTHONPATH", "").split(os.pathsep)
    if working_dir not in python_paths:
        python_paths.append(working_dir)
        os.environ["PYTHONPATH"] = os.pathsep.join(python_paths)

    # Seed for determinism. This seeds torch, numpy and python random modules
    # taking global rank into account (for multi-process distributed setting).
    # Additionally, this auto-adds a worker_init_fn to train_dataloader that
    # initializes the seed taking worker_id into account per dataloading worker
    # (see `pl_worker_init_fn()`).
    pl.seed_everything(config.seed, workers=True)

    # Dataset session paths
    def _full_paths(root: str, dataset: ListConfig) -> List[Path]:
        sessions = [session["session"] for session in dataset]
        return [Path(root).joinpath(f"{session}.hdf5") for session in sessions]

    train_sessions = _full_paths(config.dataset.root, config.dataset.train)
    val_sessions = _full_paths(config.dataset.root, config.dataset.val)
    test_sessions = _full_paths(config.dataset.root, config.dataset.test)

    # Instantiate LightningModule
    log.info(f'Instantiating LightningModule {config.module}')
    module = instantiate(config.module,
                         optimizer=config.optimizer,
                         lr_scheduler=config.lr_scheduler,
                         _recursive_=False)
    if config.checkpoint is not None:
        log.info(f'Loading from checkpoint {config.checkpoint}')
        module = module.load_from_checkpoint(config.checkpoint,
                                             optimizer=config.optimizer,
                                             lr_scheduler=config.lr_scheduler)

    # Instantiate LightningDataModule
    log.info(f'Instantiating LightningDataModule {config.datamodule}')
    datamodule = instantiate(config.datamodule,
                             batch_size=config.batch_size,
                             num_workers=config.num_workers,
                             train_sessions=train_sessions,
                             val_sessions=val_sessions,
                             test_sessions=test_sessions)

    # Instantiate transforms
    def _build_transform(configs: Sequence[DictConfig]) -> Transform[Any, Any]:
        return transforms.Compose([instantiate(cfg) for cfg in configs])

    datamodule.train_transforms = _build_transform(config.transforms.train)
    datamodule.val_transforms = _build_transform(config.transforms.val)
    datamodule.test_transforms = _build_transform(config.transforms.test)

    # Instantiate callbacks
    callback_configs = config.get("callbacks", [])
    callbacks = [instantiate(cfg) for cfg in callback_configs]

    # Train
    trainer = pl.Trainer(**config.trainer, callbacks=callbacks)
    if config.train:
        trainer.fit(module, datamodule)

    # Val and test on best model
    val_metrics = trainer.validate(module, datamodule)
    test_metrics = trainer.test(module, datamodule)

    results = {
        'val_metrics': val_metrics,
        'test_metrics': test_metrics,
        'best_model_path': trainer.checkpoint_callback.best_model_path,
    }
    pprint.pprint(results, sort_dicts=False)


if __name__ == "__main__":
    OmegaConf.register_new_resolver("cpus_per_task", utils.cpus_per_task)
    main()

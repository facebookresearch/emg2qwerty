# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import logging
import os
import pprint
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import hydra
import pytorch_lightning as pl
from hydra.utils import get_original_cwd, instantiate
from omegaconf import DictConfig, ListConfig, OmegaConf
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from torch import set_float32_matmul_precision

from emg2qwerty import transforms, utils
from emg2qwerty.lightning import AutoencoderModule
from emg2qwerty.transforms import Transform

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../config", config_name="autoencoder")
def main(config: DictConfig):
    set_float32_matmul_precision("high")
    log.info(f"\nConfig:\n{OmegaConf.to_yaml(config)}")

    # Add working dir to PYTHONPATH
    working_dir = get_original_cwd()
    python_paths = os.environ.get("PYTHONPATH", "").split(os.pathsep)
    if working_dir not in python_paths:
        python_paths.append(working_dir)
        os.environ["PYTHONPATH"] = os.pathsep.join(python_paths)

    # Create output directory for logs and plots
    output_dir = Path.cwd().joinpath("outputs")
    output_dir.mkdir(exist_ok=True)

    # Save config to output directory
    with open(output_dir.joinpath("config.yaml"), "w") as f:
        f.write(OmegaConf.to_yaml(config))

    # Seed for determinism
    pl.seed_everything(config.seed, workers=True)

    # Helper to instantiate full paths for dataset sessions
    def _full_session_paths(dataset: ListConfig) -> list[Path]:
        sessions = [session["session"] for session in dataset]
        users = [session["user"] for session in dataset]
        if config.reduced:
            return [
                Path(config.dataset.root).joinpath(f"{user}_processed").joinpath(f"{session}.hdf5")
                for session, user in zip(sessions, users)
            ]
        else:
            return [Path(config.dataset.root).joinpath(f"{session}.hdf5") for session in sessions]

    # Helper to instantiate transforms
    def _build_transform(configs: Sequence[DictConfig]) -> Transform[Any, Any]:
        return transforms.Compose([instantiate(cfg) for cfg in configs])

    # Instantiate AutoencoderModule
    log.info("Instantiating AutoencoderModule")
    module = AutoencoderModule(
        in_channels=config.autoencoder.in_channels,
        bottleneck_channels=config.autoencoder.bottleneck_channels,
        lr=config.autoencoder.lr,
    )

    if config.checkpoint is not None:
        log.info(f"Loading module from checkpoint {config.checkpoint}")
        module = module.load_from_checkpoint(
            config.checkpoint,
            in_channels=config.autoencoder.in_channels,
            bottleneck_channels=config.autoencoder.bottleneck_channels,
            lr=config.autoencoder.lr,
        )

    # Instantiate LightningDataModule
    log.info(f"Instantiating LightningDataModule {config.datamodule}")
    datamodule = instantiate(
        config.datamodule,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        train_sessions=_full_session_paths(config.dataset.train),
        val_sessions=_full_session_paths(config.dataset.val),
        test_sessions=_full_session_paths(config.dataset.test),
        train_transform=_build_transform(config.transforms.train),
        val_transform=_build_transform(config.transforms.val),
        test_transform=_build_transform(config.transforms.test),
        _convert_="object",
    )

    # Instantiate callbacks
    callback_configs = config.get("callbacks", [])
    callbacks: list[pl.Callback] = []

    # Extract model name for checkpoint naming
    model_name = "autoencoder"
    log.info(f"Using model: {model_name}")

    # Process callbacks and customize ModelCheckpoint if present
    for cfg in callback_configs:
        if cfg._target_ == "pytorch_lightning.callbacks.ModelCheckpoint":
            # Customize the ModelCheckpoint callback
            checkpoint_callback = instantiate(
                cfg,
                filename=f"{model_name}"
                + "-{epoch:02d}-{"
                + config.monitor_metric.replace("/", "_")
                + ":.4f}",
                dirpath=f"{Path.cwd()}/checkpoints/{model_name}",
            )
            callbacks.append(checkpoint_callback)
        else:
            callbacks.append(instantiate(cfg))

    # Add CSV logger to save metrics
    csv_logger = pl.loggers.CSVLogger(save_dir=output_dir, name="logs")

    # Initialize trainer
    trainer = pl.Trainer(
        **config.trainer,
        callbacks=callbacks,
        logger=[
            TensorBoardLogger(save_dir=f"{Path.cwd()}/logs/", name=model_name),
            CSVLogger(save_dir=f"{Path.cwd()}/logs/", name=model_name),
        ],
    )

    if config.train:
        # Check if a past checkpoint exists to resume training from
        checkpoint_dir = Path.cwd().joinpath("checkpoints")
        resume_from_checkpoint = utils.get_last_checkpoint(checkpoint_dir)
        if resume_from_checkpoint is not None:
            log.info(f"Resuming training from checkpoint {resume_from_checkpoint}")

        # Train
        trainer.fit(module, datamodule, ckpt_path=resume_from_checkpoint)

        # Load best checkpoint
        module = module.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

    # Validate and test on the best checkpoint (if training), or on the
    # loaded `config.checkpoint` (otherwise)
    val_metrics = trainer.validate(module, datamodule)
    test_metrics = trainer.test(module, datamodule)

    results = {
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "best_checkpoint": trainer.checkpoint_callback.best_model_path
        if hasattr(trainer, "checkpoint_callback")
        else None,
    }
    pprint.pprint(results, sort_dicts=False)

    # Save results to file
    with open(output_dir.joinpath("results.txt"), "w") as f:
        f.write(pprint.pformat(results, sort_dicts=False))

    # Plot training and validation loss if training was performed
    if config.train and csv_logger.experiment is not None:
        try:
            import matplotlib.pyplot as plt
            import pandas as pd

            metrics_file = csv_logger.experiment.metrics_file_path
            if os.path.exists(metrics_file):
                df = pd.read_csv(metrics_file)

                # Filter for loss metrics
                train_loss = df.filter(regex=r"train/loss.*$")
                val_loss = df.filter(regex=r"val/loss.*$")

                if not train_loss.empty or not val_loss.empty:
                    plt.figure(figsize=(10, 6))

                    if not train_loss.empty:
                        plt.plot(train_loss.values, label="Training Loss")

                    if not val_loss.empty:
                        plt.plot(val_loss.values, label="Validation Loss")

                    plt.xlabel("Epoch")
                    plt.ylabel("Loss")
                    plt.title("Autoencoder Training and Validation Loss")
                    plt.legend()
                    plt.grid(True)

                    # Save the plot
                    plt.savefig(output_dir.joinpath("loss_plot.png"))
                    plt.close()
                    log.info(f"Loss plot saved to {output_dir.joinpath('loss_plot.png')}")
        except ImportError:
            log.warning(
                "Could not generate loss plot. Make sure pandas and matplotlib are installed."
            )
        except Exception as e:
            log.warning(f"Error generating loss plot: {e}")


if __name__ == "__main__":
    OmegaConf.register_new_resolver("cpus_per_task", utils.cpus_per_task)
    main()

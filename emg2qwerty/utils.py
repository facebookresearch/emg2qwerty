from typing import Iterator

from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch import nn


def instantiate_optimizer_and_scheduler(params: Iterator[nn.Parameter],
                                        optimizer_config: DictConfig,
                                        lr_scheduler_config: DictConfig):
    optimizer = instantiate(optimizer_config, params)
    scheduler = instantiate(lr_scheduler_config.scheduler, optimizer)
    lr_scheduler = instantiate(lr_scheduler_config, scheduler=scheduler)
    return {
        "optimizer": optimizer,
        "lr_scheduler": OmegaConf.to_container(lr_scheduler),
    }

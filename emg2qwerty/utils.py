from typing import Iterator

from hydra.utils import instantiate
from omegaconf import DictConfig
from torch import nn


def instantiate_optimizer_and_scheduler(params: Iterator[nn.Parameter],
                                        optimizer_config: DictConfig,
                                        lr_scheduler_config: DictConfig):
    optimizer = instantiate(optimizer_config, params)
    lr_scheduler = instantiate(lr_scheduler_config, optimizer)
    return [optimizer], [lr_scheduler]

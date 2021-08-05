import Levenshtein
import torch
from torchmetrics import Metric

from emg2qwerty.data import LabelData


class CharacterErrorRate(Metric):
    """Character error rate (CER) metric by computing the Levenshtein
    edit-distance between the predicted and target sequences.

    As an instance of ``torchmetric.Metric``, synchronization across all GPUs
    involved in a distributed setting is automatically performed on every call
    to ``compute()``."""
    def __init__(self) -> None:
        super().__init__()

        self.add_state("errors", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, pred_labels: torch.Tensor,
               target_labels: torch.Tensor) -> None:
        pred_str = LabelData.from_labels(pred_labels).label_str
        target_str = LabelData.from_labels(target_labels).label_str

        self.errors += Levenshtein.distance(pred_str, target_str)
        self.total += len(target_str)

    def compute(self):
        return self.errors.float() / self.total * 100.

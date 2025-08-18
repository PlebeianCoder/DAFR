__all__ = ["AdversarialGap", "AdversarialGapProperties"]

from typing import Any

from advfaceutil.benchmark.factory import default_factory
from advfaceutil.benchmark.data import BenchmarkData, DataPropertyEnum, DataBin
from advfaceutil.benchmark.statistic.base import Statistic
from scipy.special import softmax
import numpy as np


class AdversarialGapProperties(DataPropertyEnum):
    ADVERSARIAL_GAP = "adversarial_gap"


@default_factory("AdversarialGap")
class AdversarialGap(Statistic):
    def record_statistic(self, data: BenchmarkData) -> None:
        # Get the difference between the softmax of the two largest values
        sorted_logits = np.sort(softmax(data.augmented_logits[0]))
        adversarial_gap = sorted_logits[-1] - sorted_logits[-2]
        data.add_property(AdversarialGapProperties.ADVERSARIAL_GAP, adversarial_gap)

    def collate_statistics(self, data_bin: DataBin[BenchmarkData]) -> Any:
        return self._collate_list_statistics(
            data_bin, AdversarialGapProperties.ADVERSARIAL_GAP
        )

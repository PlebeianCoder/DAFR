__all__ = ["TopK", "TopKProperties"]

from typing import Any, List, Tuple

from scipy.stats import mode
import numpy as np

from advfaceutil.benchmark.args import BenchmarkArguments
from advfaceutil.benchmark.data import (
    BenchmarkData,
    DataPropertyEnum,
    DataBin,
    Accessory,
)
from advfaceutil.benchmark.statistic.base import Statistic, StatisticFactory


class TopKProperties(DataPropertyEnum):
    TOPK = "topk"
    TOPK_AUGMENTED = "topk_augmented"


class TopK(Statistic):
    class Factory(StatisticFactory):
        def __init__(self, k: int = 3):
            self.k = k

        @staticmethod
        def name() -> str:
            return "TopK"

        def construct(
            self,
            benchmark_arguments: BenchmarkArguments,
            accessory: Accessory,
        ) -> "TopK":
            return TopK(self.name(), benchmark_arguments, accessory, self.k)

    def __init__(
        self,
        name: str,
        benchmark_arguments: BenchmarkArguments,
        accessory: Accessory,
        k: int,
    ):
        super().__init__(name, benchmark_arguments, accessory)
        self.k = k

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, TopK) and self.k == other.k

    def __hash__(self) -> int:
        return hash((self.name, self.k))

    def __str__(self):
        return f"{self.name}(k={self.k})"

    def record_statistic(self, data: BenchmarkData) -> None:
        data.add_property(TopKProperties.TOPK, data.logits[0].argsort()[::-1][: self.k])
        data.add_property(
            TopKProperties.TOPK_AUGMENTED,
            data.augmented_logits.argsort()[0][::-1][: self.k],
        )

    @staticmethod
    def _column_modes(data: List[List[int]]) -> Tuple[np.ndarray, np.ndarray]:
        data = np.array(data)
        return mode(data, keepdims=False)

    def collate_statistics(self, data_bin: DataBin[BenchmarkData]) -> Any:
        top_k = data_bin.get_property(TopKProperties.TOPK)
        top_k_augmented = data_bin.get_property(TopKProperties.TOPK_AUGMENTED)

        top_k_modes, top_k_counts = self._column_modes(top_k)
        top_k_augmented_modes, top_k_augmented_counts = self._column_modes(
            top_k_augmented
        )

        return {
            "indices": top_k_modes,
            "count_in_position": top_k_counts,
            "names": [
                self._benchmark_arguments.size.class_names[i] for i in top_k_modes
            ],
            "augmented_indices": top_k_augmented_modes,
            "augmented_count_in_position": top_k_augmented_counts,
            "augmented_names": [
                self._benchmark_arguments.size.class_names[i]
                for i in top_k_augmented_modes
            ],
        }

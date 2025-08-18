__all__ = ["Flip"]

from typing import Generator

import numpy as np

from advfaceutil.benchmark.augmentation.base import Augmentation
from advfaceutil.benchmark.factory import default_factory
from advfaceutil.benchmark.data import BenchmarkData


@default_factory("Flip")
class Flip(Augmentation):
    def load_extra_data_from_data(
        self, data: BenchmarkData
    ) -> Generator[BenchmarkData, None, None]:
        flipped_data = data.copy()
        flipped_data.image = np.fliplr(flipped_data.image)
        yield flipped_data

    def extra_data_count_from_loaded_data(
        self,
        loaded_data: int,
    ) -> int:
        return loaded_data * 2

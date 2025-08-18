__all__ = ["Noise"]

import numpy as np

from advfaceutil.benchmark.args import BenchmarkArguments
from advfaceutil.benchmark.augmentation.base import Augmentation, AugmentationFactory
from advfaceutil.benchmark.data import BenchmarkData, Accessory


class Noise(Augmentation):
    class Factory(AugmentationFactory):
        def __init__(self, scale: float = 10.0) -> None:
            self.scale = scale

        @staticmethod
        def name() -> str:
            return "Noise"

        def construct(
            self,
            benchmark_arguments: BenchmarkArguments,
            accessory: Accessory,
        ) -> "Noise":
            return Noise(self.name(), benchmark_arguments, accessory, self.scale)

    def __init__(
        self,
        name: str,
        benchmark_arguments: BenchmarkArguments,
        accessory: Accessory,
        scale: float,
    ) -> None:
        super().__init__(name, benchmark_arguments, accessory)
        self.scale = scale

    def pre_augmentation_processing(self, data: BenchmarkData) -> BenchmarkData:
        data.image = np.clip(
            data.image + np.random.normal(scale=self.scale, size=data.image.shape),
            0,
            255,
        ).astype(np.uint8)

        return data

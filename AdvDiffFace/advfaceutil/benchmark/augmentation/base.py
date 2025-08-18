__all__ = ["Augmentation", "AugmentationFactory"]

from abc import ABCMeta, abstractmethod
from typing import Generator, ClassVar, Type

from advfaceutil.benchmark.args import BenchmarkArguments
from advfaceutil.benchmark.factory import Factory, FactoryInstance
from advfaceutil.benchmark.data import BenchmarkData, Accessory


class AugmentationFactory(Factory, metaclass=ABCMeta):
    @abstractmethod
    def construct(
        self,
        benchmark_arguments: BenchmarkArguments,
        accessory: Accessory,
    ) -> "Augmentation":
        pass


class Augmentation(FactoryInstance):
    Factory: ClassVar[Type[AugmentationFactory]]

    def load_extra_data(self) -> Generator[BenchmarkData, None, None]:
        pass

    def extra_data_count(self) -> int:
        return 0

    def load_extra_data_from_data(
        self, data: BenchmarkData
    ) -> Generator[BenchmarkData, None, None]:
        pass

    def extra_data_count_from_loaded_data(self, loaded_data: int) -> int:
        return 0

    def pre_augmentation_processing(self, data: BenchmarkData) -> BenchmarkData:
        return data

    def post_augmentation_processing(self, data: BenchmarkData) -> BenchmarkData:
        return data

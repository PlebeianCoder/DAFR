__all__ = ["Factory", "FactoryInstance", "default_factory", "construct_all"]

from abc import ABCMeta, abstractmethod
from typing import TypeVar, Generic, Type, List

from advfaceutil.benchmark.args import BenchmarkArguments
from advfaceutil.benchmark.data import Accessory
from advfaceutil.utils import NamedSubType


T = TypeVar("T", bound="FactoryInstance")


class Factory(Generic[T], NamedSubType, metaclass=ABCMeta):
    @abstractmethod
    def construct(
        self,
        benchmark_arguments: BenchmarkArguments,
        accessory: Accessory,
    ) -> T:
        pass


class FactoryInstance:
    def __init__(
        self,
        name: str,
        benchmark_arguments: BenchmarkArguments,
        accessory: Accessory,
    ):
        self.name = name
        self._benchmark_arguments = benchmark_arguments
        self._accessory = accessory

    def __str__(self) -> str:
        return self.name


def default_factory(name: str, *args, **kwargs):
    def decorator(cls: Type[T]):
        class DefaultFactory(Factory[T]):
            @staticmethod
            def name() -> str:
                return name

            def construct(
                self, benchmark_arguments: BenchmarkArguments, accessory: Accessory
            ) -> T:
                return cls(name, benchmark_arguments, accessory, *args, **kwargs)

        cls.Factory = DefaultFactory
        return cls

    return decorator


def construct_all(
    factories: List[Factory[T]],
    benchmark_arguments: BenchmarkArguments,
    accessory: Accessory,
) -> List[T]:
    return [factory.construct(benchmark_arguments, accessory) for factory in factories]

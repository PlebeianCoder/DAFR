__all__ = [
    "Statistic",
    "StatisticFactory",
    "collate_statistics_for_bin",
    "AccessoryStatistic",
]

from abc import ABCMeta, abstractmethod
from typing import ClassVar, Any, TypeVar, Union, Generic, Dict, Type, List, Set

import numpy as np

from advfaceutil.benchmark.factory import Factory, FactoryInstance
from advfaceutil.benchmark.args import BenchmarkArguments
from advfaceutil.benchmark.data import (
    BenchmarkData,
    DataProperty,
    DataBin,
    Accessory,
    DataHolder,
)
from advfaceutil.utils import NamedSubType


D = TypeVar("D", bound=DataHolder)


class BaseStatistic(Generic[D], metaclass=ABCMeta):
    def __init__(self, name: str):
        self.statistic_name = name

    @abstractmethod
    def record_statistic(self, data: D) -> None:
        raise NotImplementedError

    @abstractmethod
    def collate_statistics(self, data_bin: DataBin[D]) -> Any:
        raise NotImplementedError

    @staticmethod
    def _collate_list_statistics(
        data_bin: DataBin[D], key: DataProperty, ignore_none: bool = False
    ) -> Any:
        values = data_bin.get_property(key, ignore_none)
        data = np.array(values)

        if data.size == 0:
            return {}
        elif data.size == 1:
            return data[0]

        return {
            f"mean": np.mean(data),
            f"std": np.std(data),
            f"max": np.max(data),
            f"min": np.min(data),
        }

    def __eq__(self, other: Any) -> bool:
        return (
            isinstance(other, BaseStatistic)
            and self.statistic_name == other.statistic_name
        )

    def __hash__(self) -> int:
        return hash(self.statistic_name)


class StatisticFactory(Factory, metaclass=ABCMeta):
    @abstractmethod
    def construct(
        self,
        benchmark_arguments: BenchmarkArguments,
        accessory: Accessory,
    ) -> "Statistic":
        pass


class Statistic(FactoryInstance, BaseStatistic[BenchmarkData], metaclass=ABCMeta):
    Factory: ClassVar[Type[StatisticFactory]]

    def __init__(
        self, name: str, benchmark_arguments: BenchmarkArguments, accessory: Accessory
    ):
        FactoryInstance.__init__(self, name, benchmark_arguments, accessory)
        BaseStatistic.__init__(self, name)


def collate_statistics_for_bin(
    statistics: Union[List[BaseStatistic[D]], Set[BaseStatistic[D]]],
    data_bin: DataBin[D],
) -> Dict[str, Any]:
    result = {}
    for statistic in statistics:
        result[statistic.statistic_name] = statistic.collate_statistics(data_bin)
    result["total"] = len(data_bin)
    return result


class AccessoryStatistic(BaseStatistic[Accessory], NamedSubType, metaclass=ABCMeta):
    def __init__(self):
        super().__init__(self.name())

__all__ = ["DataBin", "construct_bins_from_properties", "get_bin_file_name"]

from typing import Any, Optional, Iterable, TypeVar, Generic, List, Tuple

from advfaceutil.benchmark.data.base import DataHolder
from advfaceutil.benchmark.data.property import DataProperty

D = TypeVar("D", bound=DataHolder)


class DataBin(List[D], Generic[D]):
    """
    A data bin contains all data items associated with a specific set of properties
    """

    def __init__(self, properties: Tuple[Any, ...], entries: Iterable[D]):
        super().__init__(entries)
        self._properties = properties

    def get_property(self, key: DataProperty, ignore_none: bool = False) -> List[Any]:
        return self._get_property(key, self, ignore_none)

    @staticmethod
    def _get_property(
        key: DataProperty,
        data: List[D],
        ignore_none: bool = False,
    ) -> List[Any]:
        if ignore_none:
            return [
                entry.get_property(key)
                for entry in data
                if entry.get_property(key) is not None
            ]
        return [entry.get_property(key) for entry in data]

    def has_property(self, key: DataProperty) -> bool:
        return all(entry.has_property(key) for entry in self)

    def max_property(self, key: DataProperty) -> Any:
        return max(entry.get_property(key) for entry in self)

    def min_property(self, key: DataProperty) -> Any:
        return min(entry.get_property(key) for entry in self)

    @property
    def properties(self) -> Tuple[Any, ...]:
        return self._properties


def _bin_properties_match(
    data: D,
    bin_properties: Tuple[Any, ...],
    properties: Iterable[DataProperty],
) -> bool:
    for i, prop in enumerate(properties):
        data_val = data.get_property(prop)
        bin_val = bin_properties[i]

        if not prop.matches(data_val, bin_val):
            return False
    return True


def construct_bins_from_properties(
    data: List[D], properties: Iterable[DataProperty]
) -> List[DataBin]:
    bins = {}
    for entry in data:
        # Find the bin that should contain this entry
        matching_bin: Optional[DataBin] = None
        for bin_properties in bins.keys():
            if _bin_properties_match(entry, bin_properties, properties):
                matching_bin = bins[bin_properties]
                break

        if matching_bin is None:
            data_bin = DataBin(
                tuple(prop.value(entry.get_property(prop)) for prop in properties),
                [entry],
            )
            bins[data_bin.properties] = data_bin
        else:
            matching_bin.append(entry)

    return list(bins.values())


def get_bin_file_name(data_bin: DataBin, properties: Iterable[DataProperty]):
    return "-".join(
        f"{prop.name}_{bin_prop}"
        for bin_prop, prop in zip(data_bin.properties, properties)
    )

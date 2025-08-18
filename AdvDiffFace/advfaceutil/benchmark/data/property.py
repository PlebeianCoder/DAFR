__all__ = [
    "DataProperty",
    "SizedBinnedDataProperty",
    "SignBinnedDataProperty",
    "DataPropertyEnum",
]

from typing import TypeVar, Union, Optional
from enum import Enum

T = TypeVar("T")
N = TypeVar("N", float, int)


class DataProperty:
    def __init__(self, name: str):
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def __repr__(self):
        return f'<ImageProperty "{self}">'

    def __str__(self) -> str:
        return self._name

    def __eq__(self, other):
        return isinstance(other, DataProperty) and self._name == other._name

    def __hash__(self):
        return hash(self._name)

    def matches(self, a: T, b: T) -> bool:
        return a == b

    def value(self, value: T) -> T:
        return value

    def with_sized_bin(self, bin_size: Union[int, float]) -> "SizedBinnedDataProperty":
        return SizedBinnedDataProperty(self._name, bin_size)

    def with_sign_bin(self) -> "SignBinnedDataProperty":
        return SignBinnedDataProperty(self._name)


class SizedBinnedDataProperty(DataProperty):
    def __init__(self, name: str, bin_size: Union[int, float]):
        super().__init__(name)
        self.bin_size = bin_size

    def matches(self, a: Optional[N], b: Optional[N]) -> bool:
        if a is None or b is None:
            return False

        return round(a / self.bin_size) == round(b / self.bin_size)

    def value(self, value: Optional[N]) -> Optional[N]:
        if value is None:
            return value

        return round(value / self.bin_size) * self.bin_size


class SignBinnedDataProperty(DataProperty):
    def matches(self, a: Optional[N], b: Optional[N]) -> bool:
        if a is None or b is None:
            return False

        return (a >= 0 and b >= 0) or (a <= 0 and b <= 0)

    def value(self, value: Optional[T]) -> Optional[T]:
        if value is None:
            return value

        return 1 if value > 0 else -1 if value < 0 else 0


class DataPropertyEnum(DataProperty, Enum):
    pass

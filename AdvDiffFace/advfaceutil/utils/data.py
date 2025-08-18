import json
from abc import ABCMeta
from abc import abstractmethod
from dataclasses import asdict
from dataclasses import is_dataclass
from enum import Enum
from functools import singledispatch
from inspect import isabstract
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Type
from typing import TypeVar
from typing import Union

import numpy as np
from typing_extensions import Self

T = TypeVar("T")


def split_data(
    data: List[T],
    ratio: Optional[float] = None,
    training_limit: Optional[int] = None,
    testing_limit: Optional[int] = None,
) -> Tuple[List[T], List[T]]:
    """
    Split data into a training and testing set based on the ratio and training and testing limits.
    If the ratio and limits are not specified, then this will split the data in two.
    If the ratio and a limit is specified then the minimum of the ratio and the limit will be used.
    If one of the limits is not specified, and we have a ratio then we use the ratio to calculate the
    other limit.

    If the ratio and limits are not specified then the data is split in two.
    >>> data = list(range(10))
    >>> split_data(data)
    ([0, 1, 2, 3, 4], [5, 6, 7, 8, 9])

    If we specify the ratio the data will be split accordingly.
    >>> data = list(range(10))
    >>> split_data(data, ratio=0.3)
    ([0, 1, 2], [3, 4, 5, 6, 7, 8, 9])

    If we specify only the limits then we group based on those.
    >>> data = list(range(10))
    >>> split_data(data, training_limit=3, testing_limit=7)
    ([0, 1, 2], [3, 4, 5, 6, 7, 8, 9])

    Note that the training limit takes precedence over the testing limit.
    >>> data = list(range(10))
    >>> split_data(data, training_limit=5, testing_limit=7)
    ([0, 1, 2, 3, 4], [5, 6, 7, 8, 9])

    If we specify only one limit then the other is inferred and all the data is used.
    >>> data = list(range(10))
    >>> split_data(data, training_limit=3)
    ([0, 1, 2], [3, 4, 5, 6, 7, 8, 9])
    >>> data = list(range(10))
    >>> split_data(data, testing_limit=3)
    ([0, 1, 2, 3, 4, 5, 6], [7, 8, 9])

    If we specify the ratio and the limits then the smallest between the two are used.
    >>> data = list(range(10))
    >>> split_data(data, ratio=0.3, training_limit=4, testing_limit=7)
    ([0, 1, 2], [3, 4, 5, 6, 7, 8, 9])
    >>> data = list(range(10))
    >>> split_data(data, ratio=0.3, training_limit=4, testing_limit=3)
    ([0, 1, 2], [3, 4, 5])

    If we specify the ratio and only one limit, then the other limit is inferred using the ratio.
    >>> data = list(range(10))
    >>> split_data(data, ratio=0.3, training_limit=4)
    ([0, 1, 2], [3, 4, 5, 6, 7, 8, 9])
    >>> data = list(range(10))
    >>> split_data(data, ratio=0.3, training_limit=2)
    ([0, 1], [2, 3, 4, 5, 6, 7, 8])
    >>> split_data(data, ratio=0.3, testing_limit=2)
    ([0, 1, 2], [3, 4])

    :param data: The data to split.
    :param ratio: The ratio of training vs testing data. If not specified then the limits will be used.
    :param training_limit: The maximum size of the training set.
    :param testing_limit: The maximum size of the testing set.
    :return: The split data.
    """
    # If we have not specified the ratio or the training or testing limit then
    # we simply split the data in half
    if ratio is None and training_limit is None and testing_limit is None:
        ratio = 0.5

    # If we don't have a ratio then we must have either the training limit or the testing limit
    # (otherwise we would have set the ratio with the above condition)
    if ratio is None:
        # If the training limit is None then we must have the testing limit
        if training_limit is None:
            training_limit = len(data) - testing_limit
        # If the testing limit is None then we must have the training limit
        elif testing_limit is None:
            testing_limit = len(data) - training_limit
    else:
        # Otherwise if we have a ratio, then we use that to work out the training and testing limits
        if training_limit is None:
            training_limit = int(len(data) * ratio)
        else:
            training_limit = min(training_limit, int(len(data) * ratio))

        if testing_limit is None:
            testing_limit = int(len(data) * (1 - ratio))
        else:
            testing_limit = min(testing_limit, int(len(data) * (1 - ratio)))

    # Return the training and testing data
    return data[:training_limit], data[training_limit : training_limit + testing_limit]


def partition_data(
    data: List[T], *, size: Optional[int] = None, partitions: Optional[int] = None
) -> List[List[T]]:
    """
    Partition data into either sub-arrays of the given size or into partitions many "equal" sized arrays.
    Only one of size and partitions should be provided and both should be positive.

    If size is given and the size divides the array equally then the result will contain length / size many partitions.
    For example:
    >>> data = list(range(10))
    >>> partition_data(data, size=2)
    [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]

    If size is given but does not divide the array equally, then the final sub-array will be smaller to accommodate.
    To be precise, the final sub-array will contain length % size many elements.
    >>> data = list(range(10))
    >>> partition_data(data, size=3)
    [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]

    If partitions is given, and we can split the array into partitions many equally sized chunks then the result would
    be the same as using a size value of length / partitions. For example:
    >>> data = list(range(10))
    >>> partition_data(data, partitions=2)
    [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]

    If partitions is given, and we cannot split the array into partitions many equally sized chunks then the final
    length % partitions many sub-arrays will have size length // partitions + 1 and the remained will have size
    length / partitions. For example:
    >>> data = list(range(10))
    >>> partition_data(data, partitions=3)
    [[0, 1, 2], [3, 4, 5], [6, 7, 8, 9]]

    If partitions is given but the number of partitions is greater than the size of the data, then the data will be
    shared over all partitions with the remaining partitions being empty. For example:
    >>> data = list(range(2))
    >>> partition_data(data, partitions=3)
    [[0], [1], []]

    :param data: The data to partition.
    :param size: The maximum size of each partition.
    :param partitions: The maximum number of partitions.
    :return: The partitioned data.
    """

    # Ensure that we give either the size or the number of partitions but not both
    assert bool(size) ^ bool(partitions)

    if (size is not None and size <= 0) or (partitions is not None and partitions <= 0):
        return [data]

    if partitions is not None:
        # If the size of the data is smaller than the number of partitions then return empty arrays for the remaining
        # partitions and 1 value in each of the others
        if len(data) < partitions:
            partitioned_data = []

            for elem in data:
                partitioned_data.append([elem])

            for _ in range(partitions - len(data)):
                partitioned_data.append([])

            return partitioned_data

        # Calculate the size of the partitions
        size = len(data) // partitions
        # Calculate the number of partitions which will be larger than the calculated size in order to use all the data
        larger_partitions = len(data) % partitions
        smaller_partitions = partitions - larger_partitions

        # Add the partitions of the above size to the list
        partitioned_data = []
        for i in range(0, smaller_partitions * size, size):
            partitioned_data.append(data[i : i + size])

        # Calculate the index that we should begin adding the larger partitions
        starting_index = smaller_partitions * size

        # Increase the size such that:
        # larger_partitions * (size + 1) + (partitions - larger_partitions) * size = len(data)
        size += 1

        for i in range(starting_index, len(data), size):
            partitioned_data.append(data[i : i + size])

        return partitioned_data

    if size is not None:
        partitioned_data = []
        for i in range(0, len(data), size):
            partitioned_data.append(data[i : i + size])

        return partitioned_data

    return [data]


def validate_files_existence(*paths: Union[str, Path], error: bool = True) -> bool:
    """
    Validate that the given paths exist.

    :param paths: The paths to validate.
    :param error: Whether to raise an exception if one of the files does not exist.
    :return: Whether all files existed
    """
    for path in paths:
        path = Path(path)
        if not path.exists() or not path.is_file():
            if error:
                print("Expected file", path, "but was not found.")
                raise FileNotFoundError("Expected file %s but was not found." % path)
            return False

    return True


class NamedSubType(metaclass=ABCMeta):
    SUBCLASSES: Dict[str, Type[Self]] = {}

    def __init_subclass__(cls, **kwargs):
        if not isabstract(cls):
            cls.SUBCLASSES[cls.name()] = cls

    @staticmethod
    @abstractmethod
    def name() -> str:
        raise NotImplementedError

    @classmethod
    def from_name(cls, name: str) -> Optional[Type[Self]]:
        return cls.SUBCLASSES.get(name)

    def __str__(self):
        return self.name()

    def __repr__(self):
        return f"{self.name()}()"


@singledispatch
def to_serializable(val):
    if is_dataclass(val):
        return asdict(val)
    return str(val)


@to_serializable.register(np.float32)
def ts_float32(val: np.float32):
    return float(val)


@to_serializable.register(np.float64)
def ts_float64(val: np.float64):
    return float(val)


@to_serializable.register(np.int32)
def ts_int32(val: np.int32):
    return int(val)


@to_serializable.register(np.int64)
def ts_int64(val: np.int64):
    return int(val)


@to_serializable.register(np.ndarray)
def ts_ndarray(val: np.ndarray):
    return val.tolist()


@to_serializable.register(Path)
def ts_path(val: Path):
    return val.as_posix()


@to_serializable.register(NamedSubType)
def ts_named_subtype(val: NamedSubType):
    return val.name()


@to_serializable.register(Enum)
def ts_enum(val: Enum):
    return val.name


def to_pretty_json(data: Any) -> str:
    return json.dumps(data, indent=4, sort_keys=True, default=to_serializable)

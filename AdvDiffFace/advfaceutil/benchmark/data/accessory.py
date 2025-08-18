__all__ = ["Accessory"]

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Optional, List

import numpy as np

from advfaceutil.benchmark.args import AccessoryArguments
from advfaceutil.benchmark.data.base import DataHolder
from advfaceutil.utils import load_overlay


@dataclass(frozen=True)
class Accessory(DataHolder):
    # The path to the accessory to add to the images
    adversarial_accessory_path: Path
    # The path to the non-adversarial accessory
    non_adversarial_accessory_path: Path
    # The class that the accessory was designed for
    base_class: str
    # The class that we are supposed to target
    target_class: Optional[str]
    # The index of the base class
    base_class_index: int
    # The index of the target class
    target_class_index: Optional[int]

    # Optionally define the classes that we should use for universal impersonation
    universal_class_indices: Optional[List[int]] = None

    def __post_init__(self):
        # We need to set the initial value for the properties but since this is frozen, we must use the generic method
        object.__setattr__(self, "_properties", {})

    def copy(self) -> "Accessory":
        accessory = Accessory(
            self.adversarial_accessory_path,
            self.non_adversarial_accessory_path,
            self.base_class,
            self.target_class,
            self.base_class_index,
            self.target_class_index,
            self.universal_class_indices,
        )
        object.__setattr__(accessory, "_properties", self._properties.copy())
        return accessory

    @staticmethod
    def from_arguments(accessory_arguments: AccessoryArguments) -> "Accessory":
        return Accessory(
            accessory_arguments.adversarial_accessory_path,
            accessory_arguments.non_adversarial_accessory_path,
            accessory_arguments.base_class,
            accessory_arguments.target_class,
            accessory_arguments.base_class_index,
            accessory_arguments.target_class_index,
            accessory_arguments.universal_class_indices,
        )

    def __hash__(self):
        return hash(
            (
                self.base_class,
                self.target_class,
                self.adversarial_accessory_path,
                self.non_adversarial_accessory_path,
            )
        )

    def __lt__(self, other):
        return (
            isinstance(other, Accessory)
            and self.base_class < other.base_class
            and (
                self.target_class is None
                or other.target_class is None
                or self.target_class < other.target_class
            )
            and self.adversarial_accessory_path < other.adversarial_accessory_path
            and self.non_adversarial_accessory_path
            < other.non_adversarial_accessory_path
        )

    @cached_property
    def adversarial_accessory(self) -> np.ndarray:
        return load_overlay(self.adversarial_accessory_path)

    @cached_property
    def non_adversarial_accessory(self) -> np.ndarray:
        return load_overlay(self.non_adversarial_accessory_path)

    @property
    def is_impersonation(self) -> bool:
        return self.target_class is not None

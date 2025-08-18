__all__ = [
    "BenchmarkData",
    "CompressedBenchmarkData",
    "load_data",
    "data_count",
    "BenchmarkProperties",
]

from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any, Dict
from typing import Callable
from typing import Generator
from typing import Optional

import cv2
import numpy as np

from advfaceutil.benchmark.data.base import DataHolder
from advfaceutil.benchmark.data.property import DataPropertyEnum
from advfaceutil.datasets import FaceDatasets


@dataclass
class BenchmarkData(DataHolder):
    """
    A class to represent benchmark data and its associated properties.
    This contains the images that are created during processing and their respective predictions.

    :ivar image: The image data.
    :ivar class_name: The name of the class that the image belongs to.
    :ivar class_index: The index of the class that the image belongs to.
    :ivar augmented_image: The augmented image data (after the accessory has been placed).
    :ivar aligned_image: The aligned image data.
    :ivar augmented_aligned_image: The aligned augmented image data.
    :ivar predicted_class_index: The index of the class that the image was predicted to belong to.
    :ivar predicted_class_name: The name of the class that the image was predicted to belong to.
    :ivar logits: The logits of the image after passing through facial recognition.
    :ivar augmented_predicted_class_index: The index of the class that the augmented image was predicted to belong to.
    :ivar augmented_predicted_class_name: The name of the class that the augmented image was predicted to belong to.
    :ivar augmented_logits: The logits of the augmented image after passing through facial recognition.
    """

    image: np.ndarray = field(repr=False)
    class_name: str
    class_index: int
    path: Path
    augmented_image: Optional[np.ndarray] = None
    aligned_image: Optional[np.ndarray] = None
    augmented_aligned_image: Optional[np.ndarray] = None
    predicted_class_index: Optional[int] = None
    predicted_class_name: Optional[str] = None
    logits: Optional[np.ndarray] = None
    augmented_predicted_class_index: Optional[int] = None
    augmented_predicted_class_name: Optional[str] = None
    augmented_logits: Optional[np.ndarray] = None

    def __post_init__(self):
        super().__init__()

    def copy(self) -> "BenchmarkData":
        """
        :return: A deep copy of the data.
        """
        copied_image = BenchmarkData(
            self.image.copy(),
            self.class_name,
            self.class_index,
            self.path,
            self.augmented_image,
            self.aligned_image,
            self.augmented_aligned_image,
            self.predicted_class_index,
            self.predicted_class_name,
            self.logits,
            self.augmented_predicted_class_index,
            self.augmented_predicted_class_name,
            self.augmented_logits,
        )
        copied_image._properties = self._properties.copy()
        return copied_image

    def compress(self) -> "CompressedBenchmarkData":
        compressed = CompressedBenchmarkData(
            self.class_name,
            self.class_index,
            self.path.as_posix(),
            self.predicted_class_index,
            self.predicted_class_name,
            self.augmented_predicted_class_index,
            self.augmented_predicted_class_name,
        )
        # Since compressed is frozen, we must use the generic method to set the _properties value
        object.__setattr__(compressed, "_properties", self._properties)
        return compressed


@dataclass(frozen=True)
class CompressedBenchmarkData(DataHolder):
    class_name: str
    class_index: int
    path: str
    predicted_class_index: int
    predicted_class: str
    augmented_predicted_class_index: int
    augmented_predicted_class_name: str

    def __post_init__(self):
        # We need to set the initial value for the properties but since this is frozen, we must use the generic method
        object.__setattr__(self, "_properties", {})

    def to_json_dict(self) -> Dict[str, Any]:
        dictionary = asdict(self)

        properties = {}
        for prop, value in self._properties.items():
            properties[prop.name] = value

        dictionary["properties"] = properties

        return dictionary


def load_data(
    dataset: FaceDatasets,
    directory: Path,
    base_class: str,
    base_class_index: int,
    class_image_limit: Optional[int] = None,
    label: Optional[Callable[[BenchmarkData], None]] = None,
) -> Generator[BenchmarkData, None, None]:
    count = 0
    for image_path in directory.rglob("*"):
        if image_path.is_file() and dataset.image_belongs_to_class(
            image_path, base_class
        ):
            count += 1

            if class_image_limit is not None and count > class_image_limit:
                break

            image = cv2.imread(image_path.as_posix())
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            data = BenchmarkData(image, base_class, base_class_index, image_path)

            if label is not None:
                label(data)

            yield data


def data_count(
    dataset: FaceDatasets,
    directory: Path,
    base_class: str,
    class_data_limit: Optional[int] = None,
) -> int:
    count = sum(
        1
        for image_path in directory.rglob("*")
        if image_path.is_file()
        and dataset.image_belongs_to_class(image_path, base_class)
    )
    if class_data_limit is not None:
        return min(count, class_data_limit)
    return count


class BenchmarkProperties(DataPropertyEnum):
    BENCHMARK = "benchmark"
    ACCESSORY = "accessory"

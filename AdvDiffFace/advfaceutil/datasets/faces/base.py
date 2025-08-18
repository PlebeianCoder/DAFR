__all__ = ["RESEARCHERS", "set_researchers", "FaceDatasetSize", "FaceDataset"]

from abc import ABCMeta
from abc import abstractmethod
from enum import Enum
from pathlib import Path
from typing import Union, List, Optional

from advfaceutil.datasets.individual import IndividualDataset
from advfaceutil.datasets.multiclass import MulticlassDataset

RESEARCHERS = ["Name1", "Name2", "Name3"]


def set_researchers(researchers: List[str]) -> None:
    """
    Update the researcher names.

    :param researchers: The new list of researchers.
    """
    global RESEARCHERS
    RESEARCHERS = researchers


class FaceDatasetSize(Enum):
    def __init__(self, names: List[str]):
        self.dataset_names = names
        self.class_names = names + RESEARCHERS
        self.classes = len(names) + len(RESEARCHERS)
        self.researcher_names = RESEARCHERS

    @property
    def is_small(self) -> bool:
        return self.name == "SMALL"

    @property
    def is_large(self) -> bool:
        return self.name == "LARGE"

    @property
    def is_final(self) -> bool:
        return self.name == "FINAL"


class FaceDataset(MulticlassDataset, metaclass=ABCMeta):
    @staticmethod
    @abstractmethod
    def construct(
        data_directory: Union[str, Path],
        researchers_directory: Union[str, Path],
        size: FaceDatasetSize,
        class_image_limit: int,
        convert_to_bgr: bool = True,
    ) -> "FaceDataset":
        pass

    @classmethod
    def construct_individual_dataset(
        cls,
        data_directory: Union[str, Path],
        researchers_directory: Union[str, Path],
        size: FaceDatasetSize,
        class_name: str,
        class_image_limit: Optional[int] = None,
    ) -> IndividualDataset:
        if class_name not in size.dataset_names:
            data_directory = researchers_directory
        return IndividualDataset(
            data_directory,
            class_name,
            class_image_limit,
            image_belongs_to_class=cls.image_belongs_to_class,
        )

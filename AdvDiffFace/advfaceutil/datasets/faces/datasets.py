__all__ = ["FaceDatasets"]

from enum import Enum
from pathlib import Path
from typing import Literal, Union, Type, Optional

from advfaceutil.datasets.individual import IndividualDataset
from advfaceutil.datasets.faces.base import FaceDataset
from advfaceutil.datasets.faces.base import FaceDatasetSize
from advfaceutil.datasets.faces.pubfig import PubFigDataset
from advfaceutil.datasets.faces.pubfig import PubFigDatasetSize
from advfaceutil.datasets.faces.vggface2 import VGGFace2Dataset, VGGFace2DatasetSize


class FaceDatasets(Enum):
    PUBFIG = PubFigDataset, PubFigDatasetSize
    VGGFACE2 = VGGFace2Dataset, VGGFace2DatasetSize

    def __init__(self, dataset: Type[FaceDataset], size: Type[FaceDatasetSize]):
        self.dataset = dataset
        self.size = size

    def get_size(self, size: Literal["SMALL", "LARGE", "FINAL"]) -> FaceDatasetSize:
        return self.size[size]

    def construct(
        self,
        data_directory: Union[str, Path],
        researchers_directory: Union[str, Path],
        size: FaceDatasetSize,
        class_image_limit: int,
        convert_to_bgr: bool = True,
    ) -> FaceDataset:
        return self.dataset.construct(
            data_directory,
            researchers_directory,
            size,
            class_image_limit,
            convert_to_bgr,
        )

    def image_belongs_to_class(self, image_path: Path, class_name: str) -> bool:
        return self.dataset.image_belongs_to_class(image_path, class_name)

    def construct_individual_dataset(
        self,
        data_directory: Union[str, Path],
        researchers_directory: Union[str, Path],
        size: FaceDatasetSize,
        class_name: str,
        class_image_limit: Optional[int] = None,
    ) -> IndividualDataset:
        return self.dataset.construct_individual_dataset(
            data_directory, researchers_directory, size, class_name, class_image_limit
        )

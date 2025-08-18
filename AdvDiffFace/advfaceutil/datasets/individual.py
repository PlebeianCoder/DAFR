__all__ = ["IndividualDataset"]

from pathlib import Path
from typing import Optional, List, Tuple, Callable
from typing import Union

import torch
from torch.utils.data import DataLoader

from advfaceutil.datasets.base import Dataset
from advfaceutil.utils import split_data


class IndividualDataset(Dataset):
    """
    A dataset containing images for one class only.
    """

    @staticmethod
    def get_image_paths_for_class(
        data_directory: Path,
        class_name: str,
        image_belongs_to_class: Optional[Callable[[Path, str], bool]] = None,
    ) -> List[Path]:
        """
        Get all image paths for a class within a given directory.
        Only paths that include the given class name in the file name are accepted.

        :param data_directory: The directory containing the files to include.
        :param class_name: The name of the class to filter the files.
        :param image_belongs_to_class: A function to determine if an image belongs to a class.
        :return: The image paths for the given class name.
        """
        if image_belongs_to_class is None:

            def image_belongs_to_class(path: Path, cn: str) -> bool:
                return cn in path.name

        return list(
            filter(
                lambda f: f.is_file() and image_belongs_to_class(f, class_name),
                data_directory.rglob("*"),
            )
        )

    def __init__(
        self,
        data_directory: Union[str, Path],
        class_name: str,
        image_limit: Optional[int] = None,
        images: Optional[List[torch.Tensor]] = None,
        image_belongs_to_class: Optional[Callable[[Path, str], bool]] = None,
    ) -> None:
        """
        Initialise an individual dataset for the given data directory and class name with the
        optional image limit.

        :param data_directory: The directory to read images from.
        :param class_name: The name of the class to get images from the data directory for.
        :param image_limit: The maximum number of images to load for this class.
        :param images: The loaded images (optional).
        :param image_belongs_to_class: A function to determine if an image belongs to a class.
                                       Used only when the images are not provided.
        """
        super().__init__(convert_to_bgr=True)
        self.__data_directory = data_directory
        self.__class_name = class_name
        self.__image_limit = image_limit

        if images is None:
            self._load_data(
                self.get_image_paths_for_class(
                    Path(data_directory), class_name, image_belongs_to_class
                )
            )
        else:
            self.__images = images

    @property
    def image_limit(self) -> Optional[int]:
        """
        :return: The maximum number of images for this individual (or None).
        """
        return self.__image_limit

    def _load_data(self, paths: List[Path]) -> None:
        """
        Load the given images, limiting their number if necessary.

        :param paths: The paths to the images to load.
        """
        # Limit the number of images if necessary
        if self.__image_limit is not None:
            paths = paths[: self.__image_limit]

        self.__images = [self._load_image(path) for path in paths]

    def __len__(self) -> int:
        """
        :return: The length of the data.
        """
        return len(self.__images)

    def __getitem__(self, index: int) -> torch.Tensor:
        """
        Get a particular image from the dataset.

        :param index: The index of the image.
        :return: The corresponding image.
        """
        return self.__images[index]

    def split_training_testing(
        self,
        training_ratio: Optional[float] = None,
        training_image_limit: Optional[int] = 64,
        testing_image_limit: Optional[int] = 15,
    ) -> Tuple["IndividualDataset", "IndividualDataset"]:
        training, testing = split_data(
            self.__images, training_ratio, training_image_limit, testing_image_limit
        )

        return IndividualDataset(
            self.__data_directory, self.__class_name, training_image_limit, training
        ), IndividualDataset(
            self.__data_directory, self.__class_name, testing_image_limit, testing
        )

    def split_training_testing_as_loader(
        self,
        batch_size: int,
        training_ratio: Optional[float] = None,
        training_image_limit: Optional[int] = 64,
        testing_image_limit: Optional[int] = 15,
    ) -> Tuple[DataLoader, DataLoader]:
        return super().split_training_testing_as_loader(
            batch_size, training_ratio, training_image_limit, testing_image_limit
        )

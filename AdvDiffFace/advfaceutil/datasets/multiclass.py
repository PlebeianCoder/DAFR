__all__ = ["MulticlassDataset"]

from pathlib import Path
from typing import Optional, List, Tuple, Dict
from typing import Union
import re

import torch

from advfaceutil.datasets.base import Dataset
from advfaceutil.utils import split_data


class MulticlassDataset(Dataset):
    """
    A dataset for images of multiple classes.

    Note: to add data to the dataset, use the `load_data` function.
    """

    def __init__(
        self,
        class_image_limit: int,
        data: Optional[List[Tuple[torch.Tensor, int]]] = None,
        class_index_map: Optional[Dict[str, int]] = None,
        convert_to_bgr: bool = True,
    ) -> None:
        """
        Initialise a multiclass dataset with the given class image limit.
        If loading in data, both data and class_index_map must be provided.

        :param class_image_limit: The maximum number of images per class.
        :param data: The data to load for the dataset.
        :param class_index_map: The map from class name to class index.
        :param convert_to_bgr: Convert the images to BGR (default is True).
        """
        super().__init__(convert_to_bgr=convert_to_bgr)
        self.__class_image_limit = class_image_limit

        if (data is None) != (class_index_map is None):
            raise ValueError(
                "Cannot create a multiclass dataset with only one of data and class_index_map provided. Either provide both or neither."
            )

        if data is not None and class_index_map is not None:
            # Load the data from the parameters
            self._data = data
            self._class_index_map = class_index_map
            self._class_name_map = {
                index: clazz for clazz, index in class_index_map.items()
            }
            self._compute_image_counts()
        else:
            # Otherwise initialise empty data
            self._data: List[Tuple[torch.Tensor, int]] = []
            self._class_name_map: Dict[int, str] = {}
            self._class_index_map: Dict[str, int] = {}
            self._image_counts: Dict[str, int] = {}

    @staticmethod
    def image_belongs_to_class(image_path: Path, class_name: str) -> bool:
        class_name_in_file_name = (
            re.match(rf"{class_name}\d+", image_path.stem) is not None
        )
        class_name_in_parent_folder = any(
            class_name in path.stem for path in image_path.parents
        )
        return class_name_in_file_name or class_name_in_parent_folder

    def _compute_image_counts(self) -> None:
        """
        Compute the number of images for each class.
        """
        self._image_counts: Dict[str, int] = {}

        # Initialise the count for each class to 0
        for clazz in self._class_index_map.keys():
            self._image_counts[clazz] = 0

        # Add the counts for each class
        for _, index in self._data:
            self._image_counts[self._class_name_map[index]] += 1

    def load_data(self, directory: Union[str, Path], classes: List[str]) -> None:
        """
        Load images from the given directory, separating them based on their class.
        Images are grouped into the class if the start of the file name matches a class.

        :param directory: The directory containing the images to load.
        :param classes: The class names to load images for.
        """
        directory = Path(directory)

        # Load the class name map and index map
        for clazz in classes:
            # If we have not registered this class before
            if clazz not in self._class_index_map.keys():
                # Get the index of the class
                index = len(self._class_index_map)
                # Store the index in the class index map
                self._class_index_map[clazz] = index
                # Store the name in the class name map
                self._class_name_map[index] = clazz
                # Initialise the image count
                self._image_counts[clazz] = 0

        # Load each image
        for image_file in directory.rglob("*"):
            if not image_file.is_file():
                continue

            # Find the class name from the file name
            found_class = False
            clazz = None
            for clazz in classes:
                if self.image_belongs_to_class(image_file, clazz):
                    found_class = True
                    break

            # If we haven't found the class for this file then skip
            if not found_class or clazz is None:
                continue

            # If the class has too many images, skip
            if self._image_counts[clazz] >= self.__class_image_limit:
                continue

            image = self._load_image(image_file)

            # Add to the dataset
            self._image_counts[clazz] += 1
            self._data.append((image, self._class_index_map[clazz]))

    def split_training_testing(
        self,
        training_ratio: Optional[float] = None,
        training_image_limit: Optional[int] = None,
        testing_image_limit: Optional[int] = None,
    ) -> Tuple["MulticlassDataset", "MulticlassDataset"]:
        # Store the images per class
        image_classes = {}
        for image, clazz in self._data:
            if clazz in image_classes.keys():
                image_classes[clazz].append((image, clazz))
            else:
                image_classes[clazz] = [(image, clazz)]

        training_data = []
        testing_data = []

        # Split the data for each class
        for clazz, data in image_classes.items():
            training, testing = split_data(
                data, training_ratio, training_image_limit, testing_image_limit
            )
            training_data.extend(training)
            testing_data.extend(testing)

        return MulticlassDataset(
            self.__class_image_limit, training_data, self._class_index_map
        ), MulticlassDataset(
            self.__class_image_limit, testing_data, self._class_index_map
        )

    @property
    def classes(self) -> int:
        """
        :return: The number of classes for this multiclass dataset.
        """
        return len(self._image_counts)

    def __len__(self) -> int:
        """
        :return: The number of images in the dataset.
        """
        return len(self._data)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get an image and the corresponding one-hot vector representing which class this image is for.

        :param index: The index of the image and one-hot vector to load.
        :return: The image and corresponding one-hot vector.
        """
        zeros = torch.zeros(self.classes, dtype=torch.float32)
        image, clazz = self._data[index]
        zeros[clazz] = 1
        return image, zeros

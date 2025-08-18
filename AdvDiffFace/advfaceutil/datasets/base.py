__all__ = ["Dataset"]

from abc import ABCMeta
from abc import abstractmethod
from pathlib import Path
from typing import Optional, Tuple
from typing import Union
from PIL import ImageFile

import cv2
import imageio as io
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset


ImageFile.LOAD_TRUNCATED_IMAGES = True


class Dataset(TorchDataset, metaclass=ABCMeta):
    """
    A dataset that can be separated into a training and testing dataset.
    """

    def __init__(self, convert_to_bgr: bool = True) -> None:
        """
        Initialise a dataset, storing whether we convert images to BGR.

        :param convert_to_bgr: Whether to convert images to Blue Green Red (BGR) rather than the default RGB.
        """
        self.__convert_to_bgr = convert_to_bgr

    @property
    def convert_to_bgr(self) -> bool:
        """
        :return: Whether to convert images to BGR when loaded.
        """
        return self.__convert_to_bgr

    @abstractmethod
    def split_training_testing(
        self,
        training_ratio: Optional[float] = None,
        training_image_limit: Optional[int] = None,
        testing_image_limit: Optional[int] = None,
    ) -> Tuple["Dataset", "Dataset"]:
        """
        Split the dataset into a training and testing dataset based on the given ratio, training image limit and
        testing image limit. These parameters work like those found in `split_data` in utils.

        :param training_ratio: The ratio of training data to testing data.
        :param training_image_limit: The maximum number of training images.
        :param testing_image_limit: The maximum number of testing images.
        :return: The training and testing dataset.
        """
        pass

    def split_training_testing_as_loader(
        self,
        batch_size: int,
        training_ratio: Optional[float] = None,
        training_image_limit: Optional[int] = None,
        testing_image_limit: Optional[int] = None,
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Split the dataset into a training and testing dataset based on the given ratio, training image limit and
        testing image limit and convert the dataset into a data loader with the given batch size.
        Note: the ratio and limits work like those found in `split_data` in utils.

        :param batch_size: The number of images in each batch for the data loader.
        :param training_ratio: The ratio of training data to testing data.
        :param training_image_limit: The maximum number of training images.
        :param testing_image_limit: The maximum number of testing images.
        :return: The training and testing data loaders.
        """
        training, testing = self.split_training_testing(
            training_ratio, training_image_limit, testing_image_limit
        )
        return DataLoader(training, batch_size=batch_size, shuffle=True), DataLoader(
            testing, batch_size=batch_size, shuffle=False
        )

    def as_loader(self, *args, **kwargs) -> DataLoader:
        """
        Convert this dataset into a data loader, passing options to the DataLoader constructor.

        :param args: The arguments to pass to the DataLoader constructor.
        :param kwargs: The keyword arguments to pass to the DataLoader constructor.
        :return: The DataLoader representing this dataset.
        """
        return DataLoader(self, *args, **kwargs)

    def _load_image(self, path: Union[str, Path]) -> torch.Tensor:
        """
        Load an image from the given path and convert it into a PyTorch tensor with the channel first and with
        values normalised between 0 and 1.

        :param path: The path to the image to load.
        :return: The loaded image.
        """
        # Load the image
        image = io.imread(path, pilmode="RGB")

        # Convert to BGR if necessary
        if self.__convert_to_bgr:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Put the channel first
        image = np.transpose(image, (2, 0, 1))

        # Normalize the image between 0 and 1
        image = image / 255.0

        # Ensure that the image is a float32
        image = image.astype(np.float32)

        return torch.from_numpy(image)

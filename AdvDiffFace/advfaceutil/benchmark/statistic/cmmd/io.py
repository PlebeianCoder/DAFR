from itertools import chain
from logging import getLogger
from pathlib import Path
from typing import List
from typing import Union

import numpy as np
import tqdm
from PIL.Image import BICUBIC
from PIL.Image import Image
from PIL.Image import open
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from advfaceutil.benchmark.statistic.cmmd.embedding import ClipEmbeddingModel
from advfaceutil.datasets import FaceDatasets

IMAGE_EXTENSIONS = ["png", "jpg", "jpeg"]
LOGGER = getLogger("cmmd/io")


def _center_crop_and_resize(image: Image, size: int) -> Image:
    width, height = image.size
    least = min(width, height)
    top = (height - least) // 2
    left = (width - least) // 2
    box = (left, top, left + least, top + least)
    image = image.crop(box)
    # Note that the following performs antialiasing as well.
    return image.resize((size, size), resample=BICUBIC)  # pytype: disable=module-attr


def _read_image(path: Union[str, Path], size: int) -> np.ndarray:
    image = open(path).convert("RGB")
    if size > 0:
        image = _center_crop_and_resize(image, size)
    return np.asarray(image).astype(np.float32)


class CMMDDataset(Dataset):
    def __init__(
        self,
        reshape_to: int,
        paths: List[Path],
        datasets: List[FaceDatasets],
        max_count_per_class: int = -1,
    ) -> None:
        self._reshape_to = reshape_to
        self._paths = self._get_image_paths(paths, datasets, max_count_per_class)
        self._max_count_per_class = max_count_per_class

    def __len__(self) -> int:
        return len(self._paths)

    @staticmethod
    def _get_image_paths(
        paths: List[Path], datasets: List[FaceDatasets], max_count_per_class: int = -1
    ) -> List[Path]:
        image_paths = []

        class_image_count = {}
        for dataset in datasets:
            for class_name in chain(
                dataset.get_size("SMALL").class_names,
                dataset.get_size("LARGE").class_names,
            ):
                class_image_count[class_name] = 0

        image_files = chain(
            *[
                chain(
                    *(
                        [path.rglob(f"*.{extension}") for extension in IMAGE_EXTENSIONS]
                        + [
                            path.rglob(f"*.{extension.upper()}")
                            for extension in IMAGE_EXTENSIONS
                        ]
                    )
                )
                for path in paths
            ]
        )

        LOGGER.info("Beginning directory scan")

        filled_classes = 0

        for image_file in tqdm.tqdm(image_files):
            if not image_file.is_file():
                continue

            # Find the class name from the file name
            found_class = False
            clazz = None
            for clazz in class_image_count.keys():
                for dataset in datasets:
                    if dataset.image_belongs_to_class(image_file, clazz):
                        found_class = True
                        break

            # If we haven't found the class for this file then skip
            if not found_class or clazz is None:
                continue

            # If the class has too many images, skip
            if class_image_count[clazz] >= max_count_per_class:
                continue

            class_image_count[clazz] += 1

            image_paths.append(image_file)

            # If we have enough images for this class, then increase the number of filled classes
            if class_image_count[clazz] >= max_count_per_class:
                filled_classes += 1

            # If we have filled all classes then stop processing paths
            if filled_classes == len(class_image_count):
                break

        LOGGER.info("Directory scan complete")

        # Sort the paths to ensure a deterministic output
        image_paths.sort()

        return image_paths

    def __getitem__(self, index: int) -> np.ndarray:
        image_path = self._paths[index]

        return _read_image(image_path, self._reshape_to)


def compute_embeddings_for_image(
    image_path: Union[str, Path], embedding_model: ClipEmbeddingModel
) -> np.ndarray:
    """
    Computes the embedding for the given image.

    :param image_path: The image path to compute the embedding for.
    :param embedding_model: The embedding model to use.
    :return: Computed embeddings of shape (1, embedding_dim)
    """
    image = _read_image(image_path, embedding_model.input_image_size)

    # Normalise to the [0, 1] range
    image = image / 255.0

    if np.min(image) < 0 or np.max(image) > 1:
        raise ValueError(
            f"Image values are expected to be in [0, 1]. Found: [{np.min(image)},{np.max(image)}]"
        )

    image_batch = np.expand_dims(image, axis=0)

    # Compute the embeddings using a pmapped function
    embeddings = np.asarray(embedding_model.embed(image_batch))

    return embeddings


def compute_embeddings_for_dataset_directories(
    embedding_model: ClipEmbeddingModel,
    directories: List[Path],
    datasets: List[FaceDatasets],
    batch_size: int = 32,
    max_count_per_class: int = -1,
) -> np.ndarray:
    """
    Computes embeddings for the images in the given dataset directories.

    This drops the remainder of the images after batching with the provided
    batch_size to enable efficient computation on TPUs. This usually does not
    affect results assuming we have a large number of images in the directory.

    :param embedding_model: The embedding model to use.
    :param directories: The image directories containing PNGs or JPEGs.
    :param datasets: The datasets that the directories are for.
    :param batch_size: The batch size for the embedding model inference.
    :param max_count_per_class: The maximum number of images in the directories to use for each class.
    :return: Computed embeddings of shape (num_images, embedding_dim)
    """

    dataset = CMMDDataset(
        embedding_model.input_image_size, directories, datasets, max_count_per_class
    )

    count = len(dataset)

    LOGGER.info(
        "Calculating embeddings for %d images from directories %s",
        count,
        ", ".join(map(str, directories)),
    )

    data_loader = DataLoader(dataset, batch_size=batch_size)

    all_embeddings = []

    for batch in tqdm.tqdm(data_loader, total=count // batch_size):
        image_batch = batch.numpy()

        # Normalise to the [0, 1] range
        image_batch = image_batch / 255.0

        if np.min(image_batch) < 0 or np.max(image_batch) > 1:
            raise ValueError(
                f"Image values are expected to be in [0, 1]. Found: [{np.min(image_batch)},{np.max(image_batch)}]"
            )

        # Compute the embeddings using a pmapped function
        embeddings = np.asarray(embedding_model.embed(image_batch))
        # The output has shape (num_devices, batch_size, embedding_dim)

        all_embeddings.append(embeddings)

    return np.concatenate(all_embeddings, axis=0)

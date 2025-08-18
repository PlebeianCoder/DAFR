from argparse import ArgumentParser
from functools import lru_cache
from itertools import chain
from logging import getLogger
from pathlib import Path
from random import choice
from random import choices
from typing import Iterable
from typing import List
from typing import Optional
from typing import Set

import cv2
from pyfacear import OBJMeshIO

from advfaceutil.datasets import FaceDatasets
from advfaceutil.datasets import FaceDatasetSize
from advfaceutil.recognition.processing import AugmentationOptions
from advfaceutil.recognition.processing import DlibAugmentationOptions
from advfaceutil.recognition.processing import FaceProcessor
from advfaceutil.recognition.processing import FaceProcessors
from advfaceutil.recognition.processing import MediaPipeAugmentationOptions
from advfaceutil.utils import load_overlay
from advfaceutil.utils import LoadResearchers
from advfaceutil.utils import SetLogLevel

LOGGER = getLogger("survey")


@lru_cache
def paths_for_class(
    dataset: FaceDatasets, class_name: str, directory: Path
) -> Set[Path]:
    return set(
        path
        for path in directory.rglob("*")
        if dataset.image_belongs_to_class(path, class_name)
    )


def get_chosen_paths(
    classes: List[str],
    dataset: FaceDatasets,
    size: FaceDatasetSize,
    dataset_directory: Path,
    researchers_directory: Path,
) -> Iterable[Path]:
    chosen_paths = {}
    for class_name in classes:
        paths = paths_for_class(
            dataset,
            class_name,
            dataset_directory
            if class_name in size.dataset_names
            else researchers_directory,
        )
        if class_name in chosen_paths:
            paths.intersection_update(chosen_paths[class_name])

        if len(paths) == 0:
            LOGGER.warning(
                "Cannot find a suitable image for class %s. Either all the images have been used before or no "
                "images have been found. Skipping.",
                class_name,
            )
            continue

        if class_name in chosen_paths:
            chosen_paths[class_name].add(choice(list(paths)))
        else:
            chosen_paths[class_name] = {choice(list(paths))}

    return chain.from_iterable(chosen_paths.values())


def generate_augmented_images(
    face_processor: FaceProcessor,
    augmentation_options: AugmentationOptions,
    output_directory: Path,
    chosen_paths: Iterable[Path],
) -> None:
    for image_path in chosen_paths:
        image = cv2.imread(image_path.as_posix())
        if image is None:
            LOGGER.error("Failed to find image for %s", image_path)
            continue

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        faces = face_processor.detect_faces(image)
        if faces is None or len(faces) == 0:
            LOGGER.error("Failed to find a face for %s", image_path)
            continue

        for face in faces:
            image = face_processor.augment(image, augmentation_options, face)

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        output_image = output_directory / image_path.name
        cv2.imwrite(output_image.as_posix(), image)

        LOGGER.info("Generated augmented image and saved at %s", output_image)


def generate_augmented_images_from_directory(
    accessory_directory: Path,
    classes: List[str],
    face_processor_type: FaceProcessors,
    dataset: FaceDatasets,
    size: FaceDatasetSize,
    dataset_directory: Path,
    researchers_directory: Path,
    output_directory: Path,
    chosen_paths: Optional[Iterable[Path]] = None,
    choose_paths_per_accessory: bool = True,
    overlay_model: Optional[Path] = None,
) -> None:
    face_processor = face_processor_type.construct()

    if overlay_model is not None:
        overlay_model = OBJMeshIO.load(overlay_model)
        overlay_model.flip_texture_coords()

    if chosen_paths is None and not choose_paths_per_accessory:
        chosen_paths = get_chosen_paths(
            classes, dataset, size, dataset_directory, researchers_directory
        )

    for accessory_path in accessory_directory.rglob("*"):
        # Skip all directories
        if accessory_path.is_dir():
            continue

        if not accessory_path.is_file():
            LOGGER.warning("Skipping %s as it is not a file", accessory_path)
            continue

        accessory_image = load_overlay(accessory_path)
        if face_processor_type == FaceProcessors.DLIB:
            augmentation_options = DlibAugmentationOptions(accessory_image)
        else:
            augmentation_options = MediaPipeAugmentationOptions(
                accessory_image, overlay_model
            )

        if choose_paths_per_accessory and chosen_paths is None:
            accessory_chosen_paths = get_chosen_paths(
                classes, dataset, size, dataset_directory, researchers_directory
            )
        else:
            accessory_chosen_paths = chosen_paths

        sub_output_directory = output_directory / accessory_path.stem
        sub_output_directory.mkdir(parents=True, exist_ok=True)

        generate_augmented_images(
            face_processor,
            augmentation_options,
            sub_output_directory,
            accessory_chosen_paths,
        )
        LOGGER.info("Generated augmented images for accessory %s", accessory_path.stem)


def get_classes(
    classes: Optional[List[str]], size: FaceDatasetSize, n: int
) -> List[str]:
    if classes is None:
        return choices(size.class_names, k=n)
    return classes


def main():
    parser = ArgumentParser("Generate augmented images for a survey.")
    parser.add_argument(
        "accessory_directory",
        help="The directory containing the accessories to apply to faces",
    )
    parser.add_argument(
        "output_directory", help="The directory to save the augmented images"
    )
    parser.add_argument(
        "dataset_directory", help="The directory containing the dataset images"
    )
    parser.add_argument(
        "researchers_directory", help="The directory containing the researchers images"
    )
    parser.add_argument(
        "dataset",
        help="The dataset to use",
        choices=[dataset.name for dataset in FaceDatasets],
    )
    parser.add_argument(
        "size", help="The size of the dataset", choices=["SMALL", "LARGE"]
    )
    parser.add_argument(
        "-n",
        "--number",
        help="The number of classes to augment for each accessory",
        default=10,
        type=int,
    )
    parser.add_argument(
        "-c",
        "--add-class",
        help="The class names to augment for each accessory. If given, this will override the number of classes argument.",
        action="append",
    )
    parser.add_argument(
        "-sp",
        "--same-paths",
        help="Whether to use the same paths for each accessory",
        action="store_true",
    )
    parser.add_argument(
        "-p",
        "--add-path",
        help="The paths to the images to augment for each accessory. If given, this will override the number of classes and the class names arguments.",
        action="append",
    )
    parser.add_argument(
        "-fp",
        "--face-processor",
        help="Which face processor to use",
        choices=[fp.name for fp in FaceProcessors],
        default=FaceProcessors.DLIB.name,
    )
    parser.add_argument("-om", "--overlay-model", help="The overlay model to use")

    SetLogLevel.add_args(parser)
    LoadResearchers.add_args(parser)

    args = parser.parse_args()

    SetLogLevel.parse_args(args)
    LoadResearchers.parse_args(args)

    accessory_directory = Path(args.accessory_directory)
    if not accessory_directory.exists() or not accessory_directory.is_dir():
        LOGGER.error(
            "The accessory directory %s does not exist or is not a directory",
            accessory_directory,
        )
        return

    output_directory = Path(args.output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    dataset_directory = Path(args.dataset_directory)
    if not dataset_directory.exists() or not dataset_directory.is_dir():
        LOGGER.error(
            "The dataset directory %s does not exist or is not a directory",
            dataset_directory,
        )
        return

    researchers_directory = Path(args.researchers_directory)
    if not researchers_directory.exists() or not researchers_directory.is_dir():
        LOGGER.error(
            "The researchers directory %s does not exist or is not a directory",
            researchers_directory,
        )
        return

    dataset = FaceDatasets[args.dataset]
    size = dataset.get_size(args.size)

    n = args.number
    classes = get_classes(args.add_class, size, n)

    same_paths = args.same_paths

    paths = args.add_path
    if paths is not None:
        paths = [Path(path) for path in paths]

    face_processor = FaceProcessors[args.face_processor]
    overlay_model = args.overlay_model
    if overlay_model is not None:
        overlay_model = Path(overlay_model)
        if not overlay_model.exists() or not overlay_model.is_file():
            LOGGER.error(
                "The overlay model %s does not exist or is not a file", overlay_model
            )
            return

    generate_augmented_images_from_directory(
        accessory_directory,
        classes,
        face_processor,
        dataset,
        size,
        dataset_directory,
        researchers_directory,
        output_directory,
        paths,
        not same_paths,
        overlay_model,
    )
    LOGGER.info("Generated survey images")


if __name__ == "__main__":
    main()
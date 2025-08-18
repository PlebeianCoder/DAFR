from argparse import ArgumentParser
from logging import getLogger
from pathlib import Path
from typing import Optional

import cv2
from pyfacear import OBJMeshIO

from advfaceutil.recognition.processing import AugmentationOptions
from advfaceutil.recognition.processing import DlibAugmentationOptions
from advfaceutil.recognition.processing import FaceProcessor
from advfaceutil.recognition.processing import FaceProcessors
from advfaceutil.recognition.processing import MediaPipeAugmentationOptions
from advfaceutil.utils import load_overlay
from advfaceutil.utils import SetLogLevel

logger = getLogger("align")


def align_and_save(
    face_processor: FaceProcessor,
    image_path: str,
    save_path: str,
    visualisation_save_path: Optional[str] = None,
    augmented_save_path: Optional[str] = None,
    crop_size: int = 96,
    augmentation_options: Optional[AugmentationOptions] = None,
) -> bool:
    image = cv2.imread(image_path)

    if image is None:
        logger.error("Failed to find image for %s", image_path)
        return False

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    faces = face_processor.detect_faces(image)
    if faces is None or len(faces) == 0:
        logger.error("Failed to find a face for %s", image_path)
        return False

    for face in faces:
        if augmentation_options is not None:
            image = face_processor.augment(image, augmentation_options, face)

        aligned_face = face_processor.align(image, crop_size, face)
        if aligned_face is None:
            logger.error("Failed to align face for %s", image_path)
            return False

        aligned_face = cv2.cvtColor(aligned_face, cv2.COLOR_RGB2BGR)

        cv2.imwrite(save_path, aligned_face)

    if augmented_save_path is not None:
        cv2.imwrite(augmented_save_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    if visualisation_save_path is not None:
        display = image.copy()
        display = cv2.cvtColor(display, cv2.COLOR_RGB2BGR)

        display = face_processor.show_bounding_boxes(display, faces)
        display = face_processor.show_landmarks(display, faces)

        cv2.imwrite(visualisation_save_path, display)

    return True


def align_image(
    face_processor: FaceProcessor,
    input_path: Path,
    output_path: Path,
    crop_size: int = 96,
    visualise: bool = False,
    save_augmented: bool = False,
    augmentation_options: Optional[AugmentationOptions] = None,
):
    if output_path.exists():
        logger.warning("Overwriting output file '%s'", output_path)

    visualisation_output = None
    if visualise:
        visualisation_output = output_path.with_stem(
            output_path.stem + "_visualised"
        ).as_posix()
        if Path(visualisation_output).exists():
            logger.warning("Overwriting visualisation file '%s'", visualisation_output)

    augmentation_output = None
    if save_augmented:
        augmentation_output = output_path.with_stem(
            output_path.stem + "_augmented"
        ).as_posix()
        if Path(augmentation_output).exists():
            logger.warning("Overwriting augmented file '%s'", augmentation_output)

    result = align_and_save(
        face_processor,
        input_path.as_posix(),
        output_path.as_posix(),
        visualisation_output,
        augmentation_output,
        crop_size,
        augmentation_options,
    )

    # If we failed, end here
    if not result:
        return

    message = "Saved aligned image %s to %s"
    args = [input_path, output_path]

    if visualise:
        message += " and visualisation to %s"
        args.append(visualisation_output)

    if save_augmented:
        message += " and augmented image to %s"
        args.append(augmentation_output)

    logger.info(message, *args)


def align_directory(
    face_processor: FaceProcessor,
    input_path: Path,
    output_path: Path,
    crop_size: int = 96,
    visualise: bool = False,
    save_augmented: bool = False,
    augmentation_options: Optional[AugmentationOptions] = None,
):
    if not output_path.exists():
        output_path.mkdir()
    for image_path in input_path.rglob("*"):
        sub_path = Path(image_path.as_posix()[len(input_path.as_posix()) + 1 :])
        output_image_path = output_path / sub_path
        if not output_image_path.parent.exists():
            output_image_path.parent.mkdir(parents=True)

        align_image(
            face_processor,
            image_path,
            output_image_path,
            crop_size,
            visualise,
            save_augmented,
            augmentation_options,
        )


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "input", help="Specify the image or directory containing images to align"
    )
    parser.add_argument(
        "output",
        help="Specify the file name if aligning just an image or the directory if aligning a directory",
    )
    parser.add_argument(
        "-o", "--overlay", help="Specify the overlay image to place on top"
    )
    parser.add_argument(
        "-vis",
        "--visualise",
        help="Whether to output a visualisation image",
        action="store_true",
    )
    parser.add_argument(
        "-c", "--crop", help="How much to crop by default = 96", default=96, type=int
    )
    parser.add_argument(
        "-fp",
        "--face-processor",
        help="Which face processor to use",
        choices=[fp.name for fp in FaceProcessors],
        default=FaceProcessors.DLIB.name,
    )
    parser.add_argument(
        "-om",
        "--overlay-model",
        help="The path to the model to use for augmentation. "
        "This can only be used if MediaPipe is selected as the face processor.",
    )
    parser.add_argument(
        "-sau", "--save-augmented", help="Save the augmented image", action="store_true"
    )

    SetLogLevel.add_args(parser)

    args = parser.parse_args()

    SetLogLevel.parse_args(args)

    input_path = Path(args.input)
    output_path = Path(args.output)
    augmentation_options = None
    if args.overlay:
        overlay_path = Path(args.overlay)
        if not overlay_path.exists() or not overlay_path.is_file():
            logger.error(
                "Invalid overlay path '%s'. This must be a numpy or image file",
                overlay_path,
            )
            return

        overlay = load_overlay(overlay_path)
        logger.info("Loaded overlay '%s'", overlay_path)

        if args.face_processor == FaceProcessors.DLIB.name:
            augmentation_options = DlibAugmentationOptions(texture=overlay)
        else:
            augmentation_options = MediaPipeAugmentationOptions(texture=overlay)

    if args.overlay_model:
        if args.face_processor == FaceProcessors.DLIB.name:
            logger.error(
                "Cannot use an overlay model for the DLIB face processor. Switch to MEDIAPIPE."
            )
            return

        overlay_model_path = Path(args.overlay_model)
        if not overlay_model_path.exists() or not overlay_model_path.is_file():
            logger.error(
                "Invalid overlay model path '%s'. This must be an OBJ file.",
                overlay_model_path,
            )
            return

        if augmentation_options is None:
            logger.error(
                "Cannot use an overlay model without specifying the overlay texture. Please provide a texture"
                "as well as the model."
            )
            return

        augmentation_options.mesh = OBJMeshIO.load(overlay_model_path)
        augmentation_options.mesh.flip_texture_coords()

    if input_path.is_dir():
        face_processor = FaceProcessors[args.face_processor].construct()
        align_directory(
            face_processor,
            input_path,
            output_path,
            crop_size=args.crop,
            visualise=args.visualise,
            save_augmented=args.save_augmented,
            augmentation_options=augmentation_options,
        )
    elif input_path.is_file():
        if output_path.is_dir():
            logger.error(
                "Invalid output '%s'. This must be a file when the input is a file",
                output_path,
            )
            return

        face_processor = FaceProcessors[args.face_processor].construct()
        align_image(
            face_processor,
            input_path,
            output_path,
            crop_size=args.crop,
            visualise=args.visualise,
            save_augmented=args.save_augmented,
            augmentation_options=augmentation_options,
        )
    else:
        logger.error(
            "Invalid input '%s'. This must be either an image or a directory.",
            input_path,
        )


if __name__ == "__main__":
    main()

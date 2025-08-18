__all__ = [
    "FaceDetectionResult",
    "BoundingBox",
    "AugmentationOptions",
    "FaceProcessor",
    "FaceAligner",
]

from abc import ABCMeta
from abc import abstractmethod
from dataclasses import dataclass
from typing import Callable, List, Tuple
from typing import Optional
from typing import TypeVar
from typing import Union

import cv2
import numpy as np
import torch
import torch.nn as nn

T = TypeVar("T")
D = TypeVar("D", bound="FaceDetectionResult")


@dataclass(frozen=True)
class BoundingBox:
    """
    A bounding box around a face, represented by the coordinates of the top left and bottom right.

    :ivar x0:
        The coordinate of the left of the bounding box.
    :ivar y0:
        The coordinate of the top of the bounding box.
    :ivar x1:
        The coordinate of the right of the bounding box.
    :ivar y1:
        The coordinate of the bottom of the bounding box.

    """

    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def width(self) -> int:
        """
        :return: The width of the bounding box.
        """
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        """
        :return: The height of the bounding box.
        """
        return self.y1 - self.y0

    @property
    def left(self) -> int:
        """
        :return: The coordinate of the left of the bounding box .
        """
        return self.x0

    @property
    def top(self) -> int:
        """
        :return: The coordinate of the top of the bounding box.
        """
        return self.y0

    @property
    def right(self) -> int:
        """
        :return: The coordinate of the right of the bounding box.
        """
        return self.x1

    @property
    def bottom(self) -> int:
        """
        :return: The coordinate of the bottom of the bounding box.
        """
        return self.y1

    @property
    def area(self) -> int:
        """
        :return: The area of the bounding box
        """
        return self.width * self.height


@dataclass(frozen=True)
class FaceDetectionResult:
    """
    Encapsulate the result of detecting a face, storing the bounding box of the face.

    :ivar bounding_box:
        The bounding box of the face.
    """

    bounding_box: BoundingBox


@dataclass
class AugmentationOptions:
    """
    Options to control facial augmentation.
    This is a base class, so you should create the appropriate AugmentationOptions depending on the face processor.

    :ivar texture:
        The texture to apply to the face.
    """

    texture: np.ndarray


class FaceProcessor(metaclass=ABCMeta):
    """
    Contains functions to process images of faces including detecting, aligning and augmenting faces.
    """

    @staticmethod
    @abstractmethod
    def _validate_detection_result(
        detection_result: Union[FaceDetectionResult, List[FaceDetectionResult]]
    ) -> bool:
        """
        Validate that the given detection result or list of detection results is the correct type for this face
        processor.

        :param detection_result: The detection result(s) to verify.
        :return: Whether the given detection result(s) are valid for this face processor.
        """
        pass

    @staticmethod
    @abstractmethod
    def _validate_augmentation_options(options: AugmentationOptions) -> bool:
        """
        Validate that the given augmentation options are correct for this face processor.

        :param options: The augmentation options to verify.
        :return: Whether the given augmentation options are correct for this face processor.
        """
        pass

    @abstractmethod
    def detect_faces(self, image: np.ndarray) -> Optional[List[FaceDetectionResult]]:
        """
        Detect the faces in the given image and return the detection results for each face.
        If no faces are found, None is returned.

        :param image: The image that potentially contains faces.
        :return: The list of detection results or None if no faces are found.
        """
        pass

    @staticmethod
    def _map_detections(
        detections: Union[D, List[D]],
        function: Callable[[D], T],
    ) -> Union[T, List[T]]:
        """
        Map the given detections using the given mapper function.
        This will, for each detection result, pass the detection into the function and return either a list of mapped
        values if the given detections are a list or the value mapped otherwise.

        :param detections: The detection(s) to map.
        :param function: The mapper function that takes in a detection and returns the mapped value.
        :return: The mapped values from detections.
        """
        if isinstance(detections, list):
            return [function(detection) for detection in detections]
        return function(detections)

    @staticmethod
    def _reduce_detections(
        detections: Union[D, List[D]],
        initial: T,
        function: Callable[[T, D], T],
    ) -> T:
        """
        Reduce the given detections down to a singular value, starting at the initial value.
        This will, for each detection result, pass the accumulated value and the detection and expect the next
        accumulated value to be returned.

        :param detections: The detection(s) to reduce.
        :param initial: The initial value of the accumulator.
        :param function: The reduction function which takes in the accumulated value and the detection as parameters and
                         should return the next accumulated value.
        :return: The reduced value.
        """
        if isinstance(detections, list):
            for detection in detections:
                initial = function(initial, detection)
            return initial
        return function(initial, detections)

    @classmethod
    def show_bounding_boxes(
        cls,
        image: np.ndarray,
        detections: Union[FaceDetectionResult, List[FaceDetectionResult]],
        colour: Tuple[int, int, int] = (255, 0, 0),
        thickness: int = 1,
    ) -> np.ndarray:
        """
        Show the bounding boxes of each face in the given image.

        :param image: The image containing the faces.
        :param detections: The detection result(s) to draw bounding boxes for.
        :param colour: The colour of the bounding box (by default red).
        :param thickness: The thickness of the bounding box (by default 1).
        :return: The image with the annotated bounding boxes.
        """

        # Create the reduction function which will draw a rectangle for each detection result on the given image
        # and will return the updated image.
        def draw_box(img: np.ndarray, detection: FaceDetectionResult) -> np.ndarray:
            return cv2.rectangle(
                img,
                (detection.bounding_box.left, detection.bounding_box.top),
                (detection.bounding_box.right, detection.bounding_box.bottom),
                colour,
                thickness,
            )

        return cls._reduce_detections(
            detections,
            image,
            draw_box,
        )

    def show_landmarks(
        self,
        image: np.ndarray,
        detections: Union[FaceDetectionResult, List[FaceDetectionResult]],
        colour: Tuple[int, int, int] = (0, 255, 0),
        radius: int = 1,
    ) -> np.ndarray:
        """
        Show the detected face landmarks for the given detections in the given image.
        If the given detections are invalid, then the image will not be annotated.

        :param image: The image to annotate with the landmarks.
        :param detections: The detection result(s) to show the landmarks for.
        :param colour: The colour of the landmarks (default green).
        :param radius: The radius of the landmarks.
        :return: The image with the annotated landmarks.
        """
        # First validate the detections
        if not self._validate_detection_result(detections):
            return image
        return self._show_landmarks(image, detections, colour, radius)

    @abstractmethod
    def _show_landmarks(
        self,
        image: np.ndarray,
        detections: Union[FaceDetectionResult, List[FaceDetectionResult]],
        colour: Tuple[int, int, int] = (0, 255, 0),
        radius: int = 1,
    ) -> np.ndarray:
        """
        Show the detected face landmarks for the given detections in the given image.

        :param image: The image to annotate with the landmarks.
        :param detections: The detection result(s) to show the landmarks for.
        :param colour: The colour of the landmarks (default green).
        :param radius: The radius of the landmarks.
        :return: The image with the annotated landmarks.
        """
        pass

    def detect_largest_face(self, image: np.ndarray) -> Optional[FaceDetectionResult]:
        """
        Detect the largest face (by area) in the given image.
        If no faces are found, then None is returned

        :param image: The image that to detect faces in.
        :return: The largest face in the image or None.
        """
        faces = self.detect_faces(image)
        if faces is None:
            return None
        return max(faces, key=lambda face: face.bounding_box.area)

    def augment(
        self,
        image: np.ndarray,
        options: AugmentationOptions,
        detections: Optional[
            Union[FaceDetectionResult, List[FaceDetectionResult]]
        ] = None,
    ) -> Optional[np.ndarray]:
        """
        Augment the given image applying an overlay to a face.
        None will be returned if detections are not given or the detection results or the augmentation options are
        invalid.

        :param image: The image to augment.
        :param options: The augmentation options that control how each face is augmented.
        :param detections: The detection(s) found within the image or None to automatically detect faces.
        :return: The augmented image (provided the image contains faces and the detection results and augmentation
                 options are valid).
        """
        if detections is None:
            detections = self.detect_faces(image)
            if detections is None:
                return None
        if not self._validate_detection_result(detections):
            return None
        if not self._validate_augmentation_options(options):
            return None
        return self._augment(image, options, detections)

    @abstractmethod
    def _augment(
        self,
        image: np.ndarray,
        options: AugmentationOptions,
        detections: Union[FaceDetectionResult, List[FaceDetectionResult]],
    ) -> np.ndarray:
        """
        Augment the given image applying an overlay to a face.

        :param image: The image to augment.
        :param options: The augmentation options that control how each face is augmented.
        :param detections: The detection(s) found within the image.
        :return: The augmented image.
        """
        pass

    def align(
        self,
        image: np.ndarray,
        crop_size: int,
        detections: Optional[
            Union[FaceDetectionResult, List[FaceDetectionResult]]
        ] = None,
    ) -> Optional[Union[np.ndarray, List[np.ndarray]]]:
        """
        Align the given image based on the detection result and crop based on the crop size in order to pass into the
        facial recognition systems.

        :param image: The image to align.
        :param crop_size: The crop size to crop the face down to.
        :param detections: The detection(s) found within the image (or None to automatically detect the faces). If
                           multiple faces are given then this will return a list of aligned images otherwise just
                           one aligned image is returned.
        :return: The aligned image(s).
        """
        if detections is None:
            detections = self.detect_faces(image)
            if detections is None:
                return None
        if not self._validate_detection_result(detections):
            return None
        return self._align(image, crop_size, detections)

    @abstractmethod
    def _align(
        self,
        image: np.ndarray,
        crop_size: int,
        detections: Union[FaceDetectionResult, List[FaceDetectionResult]],
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Align the given image based on the detection result and crop based on the crop size in order to pass into the
        facial recognition systems.

        :param image: The image to align.
        :param crop_size: The crop size to crop the face down to.
        :param detections: The detection(s) found within the image. If multiple faces are given then this will return a
                           list of aligned images otherwise just one aligned image is returned.
        :return: The aligned image(s).
        """
        pass


class FaceAligner(nn.Module):
    def __init__(self, face_processor: FaceProcessor, crop_size: int):
        super().__init__()
        self._face_processor = face_processor
        self._crop_size = crop_size

    def forward(self, image: np.ndarray) -> np.ndarray:
        return self._face_processor.align(image, self._crop_size)

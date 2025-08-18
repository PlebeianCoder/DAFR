from __future__ import with_statement

from dataclasses import dataclass
from importlib.resources import path
from logging import getLogger
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import cv2
import dlib
import numpy as np

from advfaceutil.recognition.processing.base import AugmentationOptions
from advfaceutil.recognition.processing.base import BoundingBox
from advfaceutil.recognition.processing.base import FaceDetectionResult
from advfaceutil.recognition.processing.base import FaceProcessor
from advfaceutil.utils import has_alpha
from advfaceutil.utils import normalise_image
from advfaceutil.utils import unnormalise_image

LOGGER = getLogger("dlib_face_processing")

TEMPLATE = np.float32(
    [
        (0.0792396913815, 0.339223741112),
        (0.0829219487236, 0.456955367943),
        (0.0967927109165, 0.575648016728),
        (0.122141515615, 0.691921601066),
        (0.168687863544, 0.800341263616),
        (0.239789390707, 0.895732504778),
        (0.325662452515, 0.977068762493),
        (0.422318282013, 1.04329000149),
        (0.531777802068, 1.06080371126),
        (0.641296298053, 1.03981924107),
        (0.738105872266, 0.972268833998),
        (0.824444363295, 0.889624082279),
        (0.894792677532, 0.792494155836),
        (0.939395486253, 0.681546643421),
        (0.96111933829, 0.562238253072),
        (0.970579841181, 0.441758925744),
        (0.971193274221, 0.322118743967),
        (0.163846223133, 0.249151738053),
        (0.21780354657, 0.204255863861),
        (0.291299351124, 0.192367318323),
        (0.367460241458, 0.203582210627),
        (0.4392945113, 0.233135599851),
        (0.586445962425, 0.228141644834),
        (0.660152671635, 0.195923841854),
        (0.737466449096, 0.182360984545),
        (0.813236546239, 0.192828009114),
        (0.8707571886, 0.235293377042),
        (0.51534533827, 0.31863546193),
        (0.516221448289, 0.396200446263),
        (0.517118861835, 0.473797687758),
        (0.51816430343, 0.553157797772),
        (0.433701156035, 0.604054457668),
        (0.475501237769, 0.62076344024),
        (0.520712933176, 0.634268222208),
        (0.565874114041, 0.618796581487),
        (0.607054002672, 0.60157671656),
        (0.252418718401, 0.331052263829),
        (0.298663015648, 0.302646354002),
        (0.355749724218, 0.303020650651),
        (0.403718978315, 0.33867711083),
        (0.352507175597, 0.349987615384),
        (0.296791759886, 0.350478978225),
        (0.631326076346, 0.334136672344),
        (0.679073381078, 0.29645404267),
        (0.73597236153, 0.294721285802),
        (0.782865376271, 0.321305281656),
        (0.740312274764, 0.341849376713),
        (0.68499850091, 0.343734332172),
        (0.353167761422, 0.746189164237),
        (0.414587777921, 0.719053835073),
        (0.477677654595, 0.706835892494),
        (0.522732900812, 0.717092275768),
        (0.569832064287, 0.705414478982),
        (0.635195811927, 0.71565572516),
        (0.69951672331, 0.739419187253),
        (0.639447159575, 0.805236879972),
        (0.576410514055, 0.835436670169),
        (0.525398405766, 0.841706377792),
        (0.47641545769, 0.837505914975),
        (0.41379548902, 0.810045601727),
        (0.380084785646, 0.749979603086),
        (0.477955996282, 0.74513234612),
        (0.523389793327, 0.748924302636),
        (0.571057789237, 0.74332894691),
        (0.672409137852, 0.744177032192),
        (0.572539621444, 0.776609286626),
        (0.5240106503, 0.783370783245),
        (0.477561227414, 0.778476346951),
    ]
)

TPL_MIN, TPL_MAX = np.min(TEMPLATE, axis=0), np.max(TEMPLATE, axis=0)
MINMAX_TEMPLATE = (TEMPLATE - TPL_MIN) / (TPL_MAX - TPL_MIN)


@dataclass(frozen=True)
class DlibFaceDetectionResult(FaceDetectionResult):
    """
    Encapsulate the result of detecting a face using Dlib, storing the bounding box of the face and the
    landmarks.

    :ivar bounding_box:
        The bounding box of the face.
    :ivar landmarks:
        The landmarks detected for this face.
    """

    landmarks: np.ndarray


@dataclass
class DlibAugmentationOptions(AugmentationOptions):
    """
    Options to control facial augmentation using Dlib.

    :ivar texture:
        The texture to place on the face.
    :ivar additive_overlay:
        Whether the overlay texture should be placed in an additive manner.
    """

    additive_overlay: bool = False


class DlibFaceProcessor(FaceProcessor):
    __MIN_SIZE = 120
    INNER_EYES_AND_BOTTOM_LIP = np.array([39, 42, 57])

    # noinspection PyUnresolvedReferences
    def __init__(self) -> None:
        super().__init__()
        # Load in the face predictor model
        with path(
            "advfaceutil.recognition.processing",
            "shape_predictor_68_face_landmarks.dat",
        ) as name:
            LOGGER.info("Loading face predictor: %s", name)
            LOGGER.info("DLib using CUDA: %s", dlib.DLIB_USE_CUDA)

            self.__face_detector = dlib.get_frontal_face_detector()
            self.__landmark_predictor = dlib.shape_predictor(name.as_posix())

    @classmethod
    def _validate_detection_result(
        cls, detection_result: Union[FaceDetectionResult, List[FaceDetectionResult]]
    ) -> bool:
        return cls._reduce_detections(
            detection_result,
            True,
            lambda accumulator, detection: accumulator
            and isinstance(detection, DlibFaceDetectionResult),
        )

    @staticmethod
    def _validate_augmentation_options(options: AugmentationOptions) -> bool:
        return isinstance(options, DlibAugmentationOptions)

    def detect_faces(
        self, image: np.ndarray
    ) -> Optional[List[DlibFaceDetectionResult]]:
        # Resize if the image is too small
        if image.shape[0] < self.__MIN_SIZE or image.shape[1] < self.__MIN_SIZE:
            detections = self.__face_detector(image, 1)
        else:
            detections = self.__face_detector(image, 0)

        if len(detections) == 0:
            return None

        return [
            DlibFaceDetectionResult(
                bounding_box=BoundingBox(
                    x0=d.left(),
                    y0=d.top(),
                    x1=d.right(),
                    y1=d.bottom(),
                ),
                landmarks=np.float32(
                    [(p.x, p.y) for p in self.__landmark_predictor(image, d).parts()]
                ),
            )
            for d in detections
        ]

    def _show_landmarks(
        self,
        image: np.ndarray,
        detections: Union[DlibFaceDetectionResult, List[DlibFaceDetectionResult]],
        colour: Tuple[int, int, int] = (0, 255, 0),
        radius: int = 2,
    ) -> np.ndarray:
        def draw_landmarks(
            img: np.ndarray, detection: DlibFaceDetectionResult
        ) -> np.ndarray:
            for landmark in detection.landmarks:
                cv2.circle(
                    img,
                    (int(landmark[0]), int(landmark[1])),
                    radius,
                    colour,
                )
            return img

        return self._reduce_detections(detections, image, draw_landmarks)

    def _augment(
        self,
        image: np.ndarray,
        options: DlibAugmentationOptions,
        detections: Union[DlibFaceDetectionResult, List[DlibFaceDetectionResult]],
    ) -> np.ndarray:
        # Ensure that the overlay and image is between 0 and 1
        overlay = normalise_image(options.texture)
        normalised_image = normalise_image(image)

        def augment(img: np.ndarray, detection: DlibFaceDetectionResult) -> np.ndarray:
            # TODO: We want to tweak this so that the face is adjusted on the aligned image and is not static
            affine_transform = cv2.getAffineTransform(
                (
                    overlay.shape[:2] * MINMAX_TEMPLATE[self.INNER_EYES_AND_BOTTOM_LIP]
                ).astype(np.float32),
                detection.landmarks[self.INNER_EYES_AND_BOTTOM_LIP],
            )

            warped_overlay = cv2.warpAffine(
                overlay, affine_transform, (image.shape[1], image.shape[0])
            )

            if options.additive_overlay:
                # If we have an alpha channel then use the alpha channel to add the two images together
                if has_alpha(warped_overlay):
                    mask = warped_overlay[:, :, 3]
                    mask = np.expand_dims(mask, axis=2)
                    img += warped_overlay[:, :, :3] * mask
                else:
                    img += warped_overlay
            else:
                # If we have an alpha channel then use the alpha channel to blend the two images together
                if has_alpha(warped_overlay):
                    mask = warped_overlay[:, :, 3]
                    mask = np.expand_dims(mask, axis=2)
                    img = img * (1 - mask) + warped_overlay[:, :, :3] * mask
                else:
                    img = np.where(warped_overlay > 0, warped_overlay, img)

            return img

        return unnormalise_image(
            self._reduce_detections(detections, normalised_image, augment)
        )

    def _align(
        self,
        image: np.ndarray,
        crop_size: int,
        detections: Union[DlibFaceDetectionResult, List[DlibFaceDetectionResult]],
    ) -> Union[np.ndarray, List[np.ndarray]]:
        def align(detection: DlibFaceDetectionResult) -> np.ndarray:
            affine_transform = cv2.getAffineTransform(
                detection.landmarks[self.INNER_EYES_AND_BOTTOM_LIP],
                crop_size * MINMAX_TEMPLATE[self.INNER_EYES_AND_BOTTOM_LIP],
            )
            return cv2.warpAffine(image, affine_transform, (crop_size, crop_size))

        return self._map_detections(detections, align)

import sys
from dataclasses import dataclass
from logging import getLogger
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import cv2
import numpy as np
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks.python.components.containers.landmark import NormalizedLandmark
from pyfacear import EffectRenderer
from pyfacear import Environment
from pyfacear import FaceGeometry
from pyfacear import landmarks_from_results
from pyfacear import Mesh
from pyfacear import OpenGLContext
from pyfacear import OriginPointLocation
from pyfacear import PerspectiveCamera
from pyfacear import Texture

from advfaceutil.recognition.processing.base import AugmentationOptions
from advfaceutil.recognition.processing.base import BoundingBox
from advfaceutil.recognition.processing.base import FaceDetectionResult
from advfaceutil.recognition.processing.base import FaceProcessor
from advfaceutil.recognition.processing.dlib import DlibFaceProcessor
from advfaceutil.recognition.processing.dlib import MINMAX_TEMPLATE

LOGGER = getLogger("mediapipe_face_processing")


@dataclass(frozen=True)
class MediaPipeFaceDetectionResult(FaceDetectionResult):
    """
    Encapsulate the result of detecting a face using MediaPipe, storing the bounding box of the face and the
    normalised landmarks.

    :ivar bounding_box:
        The bounding box of the face.
    :ivar normalised_landmarks:
        The landmarks detected for this face, normalised between 0 and 1. To unnormalise them, multiply by the width
        and height of the image they were created from.
    """

    normalised_landmarks: List[NormalizedLandmark]


@dataclass
class MediaPipeAugmentationOptions(AugmentationOptions):
    """
    Options to control facial augmentation using MediaPipe.

    :ivar texture:
        The texture to apply to the mesh if given or on the whole face otherwise.
    :ivar mesh:
        The mesh to use on the face (e.g., a face mask or glasses).
    """

    mesh: Optional[Mesh] = None


class MediaPipeFaceProcessor(FaceProcessor):
    INNER_EYES_AND_BOTTOM_LIP = np.array([133, 362, 17])

    def __init__(self) -> None:
        super().__init__()

        # Set up the face mesh method of detecting faces
        self.__face_mesh = solutions.face_mesh.FaceMesh(
            max_num_faces=2, static_image_mode="demo.py" not in sys.argv[0]
        )

        # Create the PyFaceAR related variables for augmentation.
        self.__context = OpenGLContext(1280, 720, visible=False)
        self._effect_mesh = None
        self.__output_texture = Texture.create_empty(size=(1280, 720))
        self.__input_texture = Texture.create_empty(size=(1280, 720))
        self.__environment = Environment(
            origin_point_location=OriginPointLocation.BOTTOM_LEFT_CORNER,
            perspective_camera=PerspectiveCamera(),
        )
        self.__effect_renderer = EffectRenderer(
            environment=self.__environment,
            effect_texture=Texture(np.array([[[255, 255, 255, 255]]], dtype=np.uint8)),
            effect_mesh=None,
        )

        # Store whether we have initialised the PyFaceAR variables
        self._initialised = False

    def init(self) -> None:
        """
        Initialise the PyFaceAR related variables for augmentation.
        """
        self.__context.init()
        self.__effect_renderer.init()
        self.__output_texture.init()
        self.__input_texture.init()
        self._initialised = True

    def exit(self) -> None:
        """
        Cleanup the PyFaceAR related variables for augmentation.
        """
        self._initialised = False
        self.__context.exit()
        self.__output_texture.exit()
        self.__input_texture.exit()
        self.__effect_renderer.exit()

    @classmethod
    def _validate_detection_result(
        cls, detection_result: Union[FaceDetectionResult, List[FaceDetectionResult]]
    ) -> bool:
        return cls._reduce_detections(
            detection_result,
            True,
            lambda accumulator, detection: accumulator
            and isinstance(detection, MediaPipeFaceDetectionResult),
        )

    @staticmethod
    def _validate_augmentation_options(options: AugmentationOptions) -> bool:
        return isinstance(options, MediaPipeAugmentationOptions)

    def detect_faces(
        self, image: np.ndarray
    ) -> Optional[List[MediaPipeFaceDetectionResult]]:
        detections = self.__face_mesh.process(image)
        landmarks = landmarks_from_results(detections)
        if len(landmarks) == 0:
            return None

        height, width = image.shape[:2]

        # Convert the face landmarks into bounding box coordinates
        bounding_boxes = []
        for face_landmarks in landmarks:
            # Find the minimum and maximum x and y value
            min_x = 1
            max_x = 0
            min_y = 1
            max_y = 0

            for landmark in face_landmarks:
                min_x = min(min_x, landmark.x)
                max_x = max(max_x, landmark.x)
                min_y = min(min_y, landmark.y)
                max_y = max(max_y, landmark.y)

            # Add the bounding box for these landmarks, scaling the normalised coordinates using the image width
            # and height and clamping the values to the image size
            bounding_boxes.append(
                BoundingBox(
                    x0=max(int(min_x * width), 0),
                    y0=max(int(min_y * height), 0),
                    x1=min(int(max_x * width), width),
                    y1=min(int(max_y * height), height),
                )
            )

        return [
            MediaPipeFaceDetectionResult(
                bounding_box=bounding_box,
                normalised_landmarks=face_landmarks,
            )
            for face_landmarks, bounding_box in zip(landmarks, bounding_boxes)
        ]

    def _show_landmarks(
        self,
        image: np.ndarray,
        detections: Union[
            MediaPipeFaceDetectionResult, List[MediaPipeFaceDetectionResult]
        ],
        colour: Tuple[int, int, int] = (0, 255, 0),
        radius: int = 2,
    ) -> np.ndarray:
        # noinspection PyTypeChecker,PyUnresolvedReferences
        def draw_landmarks(img: np.ndarray, detection: MediaPipeFaceDetectionResult):
            # Convert the face detection results into the form that MediaPipe can understand
            face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            face_landmarks_proto.landmark.extend(
                [
                    landmark_pb2.NormalizedLandmark(
                        x=landmark.x, y=landmark.y, z=landmark.z
                    )
                    for landmark in detection.normalised_landmarks
                ]
            )

            # Set up the tesselation style
            tesselation_style = (
                solutions.drawing_styles.get_default_face_mesh_tesselation_style()
            )
            tesselation_style.thickness = radius
            tesselation_style.circle_radius = radius

            # Draw the landmarks using the MediaPipe drawing utilities
            solutions.drawing_utils.draw_landmarks(
                image=img,
                landmark_list=face_landmarks_proto,
                connections=solutions.face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=tesselation_style,
            )
            solutions.drawing_utils.draw_landmarks(
                image=img,
                landmark_list=face_landmarks_proto,
                connections=solutions.face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=solutions.drawing_styles.get_default_face_mesh_contours_style(),
            )
            return img

        return self._reduce_detections(detections, image, draw_landmarks)

    def _augment(
        self,
        image: np.ndarray,
        options: MediaPipeAugmentationOptions,
        detections: Union[
            MediaPipeFaceDetectionResult, List[MediaPipeFaceDetectionResult]
        ],
    ) -> np.ndarray:
        # If we are not initialised then initialise PyFaceAR
        if not self._initialised:
            self.init()

        # Update the effect texture with the given texture
        self.__effect_renderer.effect_texture.update(options.texture)

        # Set the effect mesh
        self.__effect_renderer.effect_mesh = options.mesh
        # If the effect mesh is not None and is not initialised then initialise the mesh
        if options.mesh is not None and not options.mesh.initialised:
            options.mesh.init()

        height, width = image.shape[:2]

        # Resize the OpenGL context if necessary
        if width != self.__context.width or height != self.__context.height:
            self.__context.resize(width, height)
        # Resize the output texture if necessary
        if (
            width != self.__output_texture.width
            or height != self.__output_texture.height
        ):
            self.__output_texture.update(size=(width, height))

        # Update the input texture
        self.__input_texture.update(image)

        if isinstance(detections, MediaPipeFaceDetectionResult):
            detections = [detections]

        # Estimate the face geometries for each of the detected faces
        face_geometries = FaceGeometry.estimate_face_geometries(
            self.__environment,
            list(map(lambda detection: detection.normalised_landmarks, detections)),
            width,
            height,
        )

        # Draw the augmentation effect
        self.__effect_renderer.render_effect(
            face_geometries,
            width,
            height,
            self.__input_texture,
            self.__output_texture,
        )

        return self.__output_texture.read().copy()

    def _align(
        self,
        image: np.ndarray,
        crop_size: int,
        detections: Union[
            MediaPipeFaceDetectionResult, List[MediaPipeFaceDetectionResult]
        ],
    ) -> Union[np.ndarray, List[np.ndarray]]:
        # def align(detection: MediaPipeFaceDetectionResult) -> np.ndarray:
        #     # To align using MediaPipe we have to approximate which landmarks roughly align to what we use with Dlib.

        #     # Get the landmarks that we care about from MediaPipe
        #     detection_landmarks = []
        #     for index in self.INNER_EYES_AND_BOTTOM_LIP:
        #         landmark = detection.normalised_landmarks[index]
        #         detection_landmarks.append(
        #             (landmark.x * image.shape[1], landmark.y * image.shape[0])
        #         )

        #     detection_landmarks = np.array(detection_landmarks)

        #     # Warp the face based on how the MediaPipe landmarks move to the Dlib landmarks.
        #     affine_transform = cv2.getAffineTransform(
        #         detection_landmarks.astype(np.float32),
        #         crop_size
        #         * MINMAX_TEMPLATE[DlibFaceProcessor.INNER_EYES_AND_BOTTOM_LIP],
        #     )
        #     return cv2.warpAffine(image, affine_transform, (crop_size, crop_size))

        # return self._map_detections(detections, align)
        return image

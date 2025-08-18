from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from mediapipe.tasks.python.components.containers.landmark import Landmark
from mediapipe.tasks.python.components.containers.landmark import NormalizedLandmark

from ..mesh import CanonicalFaceMesh
from ..mesh import Mesh
from .camera import Environment
from .camera import PerspectiveCameraFrustum
from .converter import convert_landmarks

canonical_face_mesh = CanonicalFaceMesh.get()


@dataclass(frozen=True)
class FaceGeometry:
    """
    A class to represent the geometry of a face. This stores the mesh of landmarks and the pose transform matrix which
    can be used to transform a mesh to the pose of the face (see model_matrix in the renderer).
    """

    mesh: Mesh
    pose_transform_matrix: np.ndarray[(4, 4), np.dtype[np.float64]]

    @classmethod
    def estimate_face_geometry(
        cls,
        environment: Environment,
        face_landmarks: list[NormalizedLandmark],
        frame_width: int,
        frame_height: int,
    ) -> FaceGeometry:
        """
        Estimate the face geometry from the face landmarks.
        This converts the normalised landmarks in screen space to metric space and then creates a face geometry from
        the metric landmarks.

        :param environment: The environment to use when estimating the face geometry.
        :param face_landmarks: The face landmarks to use when estimating the face geometry.
        :param frame_width: The width of the frame.
        :param frame_height: The height of the frame.
        :return: The estimated face geometry.
        """
        perspective_camera_frustum = PerspectiveCameraFrustum(
            perspective_camera=environment.perspective_camera,
            frame_width=frame_width,
            frame_height=frame_height,
        )
        metric_face_landmarks, pose_transform_matrix = convert_landmarks(
            face_landmarks,
            perspective_camera_frustum,
            environment.origin_point_location,
        )
        return FaceGeometry(
            mesh=cls.__mesh_from_landmarks(metric_face_landmarks),
            pose_transform_matrix=pose_transform_matrix,
        )

    @classmethod
    def estimate_face_geometries(
        cls,
        environment: Environment,
        face_landmarks: list[list[NormalizedLandmark]],
        frame_width: int,
        frame_height: int,
    ) -> list["FaceGeometry"]:
        """
        Estimate the face geometry from the face landmarks.
        This converts the normalised landmarks in screen space to metric space and then creates a face geometry from
        the metric landmarks.

        :param environment: The environment to use when estimating the face geometry.
        :param face_landmarks: The face landmarks to use when estimating the face geometry.
                               This can contain multiple faces.
        :param frame_width: The width of the frame.
        :param frame_height: The height of the frame.
        :return: The estimated face geometry.
        """
        perspective_camera_frustum = PerspectiveCameraFrustum(
            perspective_camera=environment.perspective_camera,
            frame_width=frame_width,
            frame_height=frame_height,
        )

        multi_face_geometry = []

        # For each face
        for screen_face_landmarks in face_landmarks:
            # If the landmarks are too compact, skip this face
            if cls.__is_screen_landmark_list_too_compact(screen_face_landmarks):
                continue

            # Convert the landmarks to metric space
            metric_face_landmarks, pose_transform_matrix = convert_landmarks(
                screen_face_landmarks,
                perspective_camera_frustum,
                environment.origin_point_location,
            )

            # Create the face geometry from the metric landmarks
            multi_face_geometry.append(
                FaceGeometry(
                    mesh=cls.__mesh_from_landmarks(metric_face_landmarks),
                    pose_transform_matrix=pose_transform_matrix,
                )
            )

        return multi_face_geometry

    @staticmethod
    def __mesh_from_landmarks(landmarks: list[Landmark]) -> Mesh:
        """
        Create a mesh from the landmarks.
        Note: this mesh will have the same texture coordinates and indices as the canonical face mesh.

        :param landmarks: The landmarks to create the mesh from.
        :return: The mesh created from the landmarks.
        """
        # Convert the landmarks into a vertices numpy array
        vertices = []
        for landmark in landmarks:
            vertices.append([landmark.x, landmark.y, landmark.z])
        vertices = np.asarray(vertices, dtype=np.float32)

        # Construct the mesh
        return Mesh(
            vertices=vertices,
            texture_coords=canonical_face_mesh.texture_coords,
            indices=canonical_face_mesh.indices,
        )

    @staticmethod
    def __is_screen_landmark_list_too_compact(
        screen_landmarks: list[NormalizedLandmark],
    ) -> bool:
        """
        Check if the screen landmark list is too compact.
        This checks if the landmarks are too close together.

        :param screen_landmarks: The screen landmarks to check.
        :return: Whether the screen landmark list is too compact.
        """
        # Compute the mean of the x-axis and y-axis
        mean_x = 0.0
        mean_y = 0.0

        for i, landmark in enumerate(screen_landmarks):
            mean_x += (landmark.x - mean_x) / (i + 1)
            mean_y += (landmark.y - mean_y) / (i + 1)

        # Compute the maximum squared distance from the mean
        max_sq_dist = 0.0
        for landmark in screen_landmarks:
            dist_x = landmark.x - mean_x
            dist_y = landmark.y - mean_y
            max_sq_dist = max(max_sq_dist, dist_x * dist_x + dist_y * dist_y)

        # If the distance is too small then the landmarks are too compact
        return np.sqrt(max_sq_dist) <= 1e-3

from __future__ import annotations

from typing import Any

import numpy as np
from mediapipe.tasks.python.components.containers.landmark import Landmark
from mediapipe.tasks.python.components.containers.landmark import NormalizedLandmark

from ..mesh import CanonicalFaceMesh
from .camera import OriginPointLocation
from .camera import PerspectiveCameraFrustum
from .procrustes_solver import solve_weighted_orthogonal_problem

canonical_face_mesh = CanonicalFaceMesh.get()


# This file can be considered as the Python equivalent of the MediaPipe geometry_pipeline.cc file.
# Hence, we copied the comments that explain the algorithm.


def convert_landmarks(
    screen_landmark_list: list[NormalizedLandmark],
    perspective_camera_frustum: PerspectiveCameraFrustum,
    origin_point_location: OriginPointLocation,
) -> tuple[list[Landmark], np.ndarray[(4, 4), np.dtype[np.float64]]]:
    """
    Convert the landmarks from screen space to metric space and estimate the pose transformation matrix.

    :param screen_landmark_list: The list of screen space landmarks.
    :param perspective_camera_frustum: The perspective camera frustum.
    :param origin_point_location: The origin point location.
    :return: The list of metric space landmarks and the pose transformation matrix.
    """

    # When using the FaceLandmarker in MediaPipe, the landmark list can contain more landmarks than the canonical face
    # mesh vertex count, so we ensure that the landmark list is the same length as the canonical face mesh vertex count.
    screen_landmark_list = screen_landmark_list[: canonical_face_mesh.vertex_count]

    # Note: the input screen landmarks are in the left-handed coordinate system, however any metric landmarks -
    #       including the canonical metric landmarks, the final runtime metric landmarks and any intermediate runtime
    #       metric landmarks - are in the right-handed coordinate system.
    #       To keep the logic correct, the landmark set handedness is changed any time the screen-to-metric semantic
    #       barrier is passed.

    screen_landmarks = convert_landmark_list_to_matrix(screen_landmark_list)

    # 1. Project X- and Y- screen landmark coordinates at the Z near plane.
    screen_landmarks = project_xy(
        perspective_camera_frustum, origin_point_location, screen_landmarks
    )

    depth_offset = np.mean(screen_landmarks[2, :])

    # 2. Estimate a canonical-to-runtime landmark set scale by running the Procrustes solver using the runtime screen
    #    landmarks.
    #    On this iteration, screen landmarks are used instead of unprojected metric landmarks as it is not safe to
    #    unproject due to the relative nature of the input screen landmark Z coordinate.
    intermediate_landmarks = screen_landmarks.copy()
    intermediate_landmarks = change_handedness(intermediate_landmarks)

    first_iteration_scale = estimate_scale(intermediate_landmarks)

    # 3. Use the canonical-to-runtime scale from (2) to unproject the screen landmarks. The result is referenced as
    #    "intermediate landmarks" because they are the first estimation of the resulting metric landmarks, but
    #    are not quite there yet
    intermediate_landmarks = screen_landmarks.copy()
    intermediate_landmarks = move_and_rescale_z(
        perspective_camera_frustum,
        depth_offset,
        first_iteration_scale,
        intermediate_landmarks,
    )
    intermediate_landmarks = unproject_xy(
        perspective_camera_frustum, intermediate_landmarks
    )
    intermediate_landmarks = change_handedness(intermediate_landmarks)

    # 4. Estimate a canonical-to-runtime landmark set scale by running the Procrustes solver using the intermediate
    #    runtime landmarks.
    second_iteration_scale = estimate_scale(intermediate_landmarks)

    # 5. Use the product of the scale factors from (2) and (4) to unproject the screen landmarks a second time.
    #    This is the second and the final estimation of the metric landmarks.
    total_scale = first_iteration_scale * second_iteration_scale

    metric_landmarks = screen_landmarks
    metric_landmarks = move_and_rescale_z(
        perspective_camera_frustum, depth_offset, total_scale, metric_landmarks
    )
    metric_landmarks = unproject_xy(perspective_camera_frustum, metric_landmarks)
    metric_landmarks = change_handedness(metric_landmarks)

    # 6. Estimate the pose transformation matrix using the canonical face mesh and the metric landmarks
    #    This matrix will transform the canonical face mesh to the metric landmarks and can be used for face effects.
    pose_transformation_matrix = solve_weighted_orthogonal_problem(
        canonical_face_mesh.vertices.T, metric_landmarks, canonical_face_mesh.weights
    )

    # 7. Multiply each of the metric landmarks by the inverse pose transformation matrix to align the runtime metric
    #    face landmarks with the canonical metric face landmarks.
    metric_landmarks = (
        np.linalg.inv(pose_transformation_matrix)
        @ np.vstack((metric_landmarks, np.ones((1, metric_landmarks.shape[1]))))
    )[0:3, :]

    return convert_matrix_to_landmark_list(metric_landmarks), pose_transformation_matrix


def project_xy(
    pcf: PerspectiveCameraFrustum,
    origin_point_location: OriginPointLocation,
    landmarks: np.ndarray[(3, Any), np.dtype[np.float64]],
) -> np.ndarray[(3, Any), np.dtype[np.float64]]:
    """
    Project X- and Y- coordinates of the landmarks at the Z near plane.

    :param pcf: The perspective camera frustum.
    :param origin_point_location: The origin point location.
    :param landmarks: The landmarks to project.
    :return: The projected landmarks.
    """
    x_scale = pcf.right - pcf.left
    y_scale = pcf.top - pcf.bottom
    x_translation = pcf.left
    y_translation = pcf.bottom

    if origin_point_location == OriginPointLocation.TOP_LEFT_CORNER:
        landmarks[1, :] = 1.0 - landmarks[1, :]

    landmarks = landmarks * np.array([[x_scale, y_scale, x_scale]]).T
    landmarks = landmarks + np.array([[x_translation, y_translation, 0.0]]).T
    return landmarks


def estimate_scale(landmarks: np.ndarray[(3, Any), np.dtype[np.float64]]) -> float:
    """
    Estimate the scale of the landmarks using the Procrustes solver.

    :param landmarks: The landmarks to estimate the scale of.
    :return: The estimated scale
    """
    transformation_matrix = solve_weighted_orthogonal_problem(
        canonical_face_mesh.vertices.T, landmarks, canonical_face_mesh.weights
    )
    return np.linalg.norm(transformation_matrix[:, 0])


def move_and_rescale_z(
    pcf: PerspectiveCameraFrustum,
    depth_offset: float,
    scale: float,
    landmarks: np.ndarray[(3, Any), np.dtype[np.float64]],
) -> np.ndarray[(3, Any), np.dtype[np.float64]]:
    """
    Move and rescale the Z coordinate of the landmarks.

    :param pcf: The perspective camera frustum.
    :param depth_offset: The depth offset.
    :param scale: The scale factor.
    :param landmarks: The landmarks to move and rescale.
    :return: The moved and rescaled landmarks.
    """
    landmarks[2, :] = (landmarks[2, :] - depth_offset + pcf.near) / scale
    return landmarks


def unproject_xy(
    pcf: PerspectiveCameraFrustum, landmarks: np.ndarray[(3, Any), np.dtype[np.float64]]
) -> np.ndarray[(3, Any), np.dtype[np.float64]]:
    """
    Unproject the X- and Y- coordinates of the landmarks.

    :param pcf: The perspective camera frustum.
    :param landmarks: The landmarks to unproject.
    :return: The unprojected landmarks.
    """
    landmarks[0, :] = (landmarks[0, :] * landmarks[2, :]) / pcf.near
    landmarks[1, :] = (landmarks[1, :] * landmarks[2, :]) / pcf.near
    return landmarks


def change_handedness(
    landmarks: np.ndarray[(3, Any), np.dtype[np.float64]]
) -> np.ndarray[(3, Any), np.dtype[np.float64]]:
    """
    Change the handedness of the landmarks.
    This is needed every time the screen-to-metric semantic barrier is passed.

    :param landmarks: The landmarks to change the handedness of.
    :return: The landmarks with the handedness changed.
    """
    landmarks[2, :] *= -1.0
    return landmarks


def convert_landmark_list_to_matrix(
    landmark_list: list[NormalizedLandmark],
) -> np.ndarray[(3, Any), np.dtype[np.float64]]:
    """
    Convert a list of landmarks to a matrix.

    :param landmark_list: The list of landmarks to convert.
    :return: The resulting matrix.
    """
    return np.array(
        [(landmark.x, landmark.y, landmark.z) for landmark in landmark_list],
        dtype=np.float64,
    ).T


def convert_matrix_to_landmark_list(
    matrix: np.ndarray[(3, Any), np.dtype[np.float64]]
) -> list[Landmark]:
    """
    Convert a matrix to a list of landmarks.

    :param matrix: The matrix to convert.
    :return: The resulting list of landmarks.
    """
    landmark_list = []
    for i in range(matrix.shape[1]):
        landmark_list.append(
            Landmark(
                x=matrix[0, i].item(),
                y=matrix[1, i].item(),
                z=matrix[2, i].item(),
            )
        )
    return landmark_list

"""Landmark related functions and classes.

Notes
-----
The `landmarks_from_results` function can be used to extract and convert the normalised landmarks into a consistent
format from the results object. The results object can be from the FaceLandmarker or FaceMesh components from MediaPipe.
"""
from __future__ import annotations

from typing import Any

from mediapipe.tasks.python.components.containers.landmark import NormalizedLandmark
from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarkerResult


def landmarks_from_results(results: Any) -> list[list[NormalizedLandmark]]:
    """Extract the normalised landmarks from the results object.

    Extracts and converts the normalised landmarks into a consistent format from the results object. The results object
    can be from the FaceLandmarker or FaceMesh components from MediaPipe.

    Parameters
    ----------
    results : Any
        The results object. This must be either a FaceLandmarkerResult or a result object that contains the
        `multi_face_landmarks` attribute as returned by FaceMesh from MediaPipe.

    Returns
    -------
    list[list[NormalizedLandmark]]
        The normalised landmarks extracted from the results object.

    Raises
    ------
    ValueError
        If the landmarks cannot be extracted from the results object.

    Notes
    -----
    It is expected that this function will be used before passing in normalised landmarks to the FaceGeometry,
    `estimate_face_geometry` function.

    Examples
    --------
    Using the landmarks_from_results function with the FaceMesh component from MediaPipe:

    >>> from PIL import Image
    >>> import mediapipe as mp
    >>> import numpy as np
    >>> pil_img = Image.new("RGB", (60, 30), color="red")
    >>> with mp.solutions.face_mesh.FaceMesh() as face_mesh:
    ...     results = face_mesh.process(np.asarray(pil_img))
    >>> landmarks_from_results(results)
    []

    Using the landmarks_from_results function with the FaceLandmarker component from MediaPipe:

    >>> import mediapipe as mp
    >>> from PIL import Image
    >>> import numpy as np
    >>> pil_img = Image.new("RGB", (60, 30), color="red")
    >>> image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.asarray(pil_img))
    >>> face_landmarker = mp.tasks.vision.face_landmarker.FaceLandmarker.create_from_model_path("face_landmarker.task") # doctest: +SKIP
    >>> results = face_landmarker.detect(image) # doctest: +SKIP
    >>> landmarks_from_results(results)
    []
    """
    if isinstance(results, FaceLandmarkerResult):
        return results.face_landmarks
    if hasattr(results, "multi_face_landmarks"):
        # multi_face_landmarks can be None
        if not results.multi_face_landmarks:
            return []
        return [
            [
                NormalizedLandmark.create_from_pb2(landmark)
                for landmark in landmarks.landmark
            ]
            for landmarks in results.multi_face_landmarks
        ]
    raise ValueError(f"Unable to extract landmarks from result: {results}")

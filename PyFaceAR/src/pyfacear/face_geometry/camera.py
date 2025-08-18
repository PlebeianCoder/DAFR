"""Dataclasses corresponding to the camera that is used in face geometry estimation.

This module contains classes to represent a camera with perspective projection, and its view frustum used in the
environment for face geometry estimation.

The `PerspectiveCamera` class represents a camera with perspective projection, defined by its vertical field of view,
near and far planes. The `PerspectiveCameraFrustum` class represents the view frustum of a perspective camera, defined by
its left, right, bottom, top, near and far planes. The `Environment` class represents the environment in which face
geometry is calculated, defined by the origin point location and perspective camera.
"""
from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from dataclasses import InitVar
from enum import auto
from enum import Enum
from functools import lru_cache

import numpy as np


@dataclass(unsafe_hash=True)
class PerspectiveCamera:
    """Representation of a camera with perspective projection.

    Representation of a camera with perspective projection, defined by its vertical field of view, near and far planes.

    Attributes
    ----------
    vertical_fov_degrees : float
        The vertical field of view in degrees. By default, this is 60 degrees.
    near : float
        The distance to the near plane. By default, this is 1.0.
    far : float
        The distance to the far plane. By default, this is 10000.0.

    Examples
    --------
    >>> PerspectiveCamera()
    PerspectiveCamera(vertical_fov_degrees=60, near=1.0, far=10000.0)

    >>> PerspectiveCamera(vertical_fov_degrees=45, near=0.1, far=100.0)
    PerspectiveCamera(vertical_fov_degrees=45, near=0.1, far=100.0)
    """

    vertical_fov_degrees: float = 60
    near: float = 1.0
    far: float = 10000.0

    @lru_cache(1)
    def projection_matrix(
        self, frame_width: int, frame_height: int
    ) -> np.ndarray[(4, 4), np.dtype[np.float32]]:
        """Calculate the perspective projection matrix given the frame dimensions.

        Calculate the perspective projection matrix given the frame dimensions. The matrix is calculated using the
        vertical field of view, aspect ratio of the given dimensions and near and far planes.

        Parameters
        ----------
        frame_width : int
            The width of the frame.
        frame_height : int
            The height of the frame.

        Returns
        -------
        np.ndarray[(4, 4), np.float32]
            The computed perspective projection matrix.

        Examples
        --------
        >>> camera = PerspectiveCamera()
        >>> camera.projection_matrix(1920, 1080)
        array([[ 0.97427858,  0.        ,  0.        ,  0.        ],
               [ 0.        ,  1.73205081,  0.        ,  0.        ],
               [ 0.        ,  0.        , -1.00020002, -2.00020002],
               [ 0.        ,  0.        , -1.        ,  0.        ]])

        >>> camera = PerspectiveCamera(vertical_fov_degrees=45, near=0.1, far=100.0)
        >>> camera.projection_matrix(1920, 1080)
        array([[ 1.35799513,  0.        ,  0.        ,  0.        ],
               [ 0.        ,  2.41421356,  0.        ,  0.        ],
               [ 0.        ,  0.        , -1.002002  , -0.2002002 ],
               [ 0.        ,  0.        , -1.        ,  0.        ]])

        Notes
        -----
        The matrix is calculated using the following formula:

        .. math::
            \\begin{bmatrix}
            \\frac{1}{\\text{aspect} \\times \\tan(\\frac{\\text{fov}}{2})} & 0 & 0 & 0 \\\\
            0 & \\frac{1}{\\tan(\\frac{\\text{fov}}{2})} & 0 & 0 \\\\
            0 & 0 & \\frac{\\text{far} + \\text{near}}{\\text{near} - \\text{far}} & \\frac{2 \\times \\text{far} \\times \\text{near}}{\\text{near} - \\text{far}} \\\\
            0 & 0 & -1 & 0
            \\end{bmatrix}

        where :math:`\\text{aspect} = \\frac{\\text{frame_width}}{\\text{frame_height}}`.

        Additionally, the result of this function is cached using the `lru_cache` decorator to avoid recalculating the
        matrix when the same frame dimensions are used.
        """
        aspect = frame_width / frame_height
        f = 1.0 / np.tan(0.5 * np.radians(self.vertical_fov_degrees))

        return np.array(
            [
                [f / aspect, 0, 0, 0],
                [0, f, 0, 0],
                [
                    0,
                    0,
                    (self.far + self.near) / (self.near - self.far),
                    (2 * self.far * self.near) / (self.near - self.far),
                ],
                [0, 0, -1, 0],
            ]
        )


@dataclass
class PerspectiveCameraFrustum:
    """View frustum of a perspective camera.

    View frustum of a perspective camera, defined by its left, right, bottom, top, near and far planes.

    Attributes
    ----------
    left : float
        The distance to the left plane.
    right : float
        The distance to the right plane.
    bottom : float
        The distance to the bottom plane.
    top : float
        The distance to the top plane.
    near : float
        The distance to the near plane.
    far : float
        The distance to the far plane.

    Examples
    --------
    >>> camera = PerspectiveCamera()
    >>> frustum = PerspectiveCameraFrustum(perspective_camera=camera, frame_width=1920, frame_height=1080)
    >>> frustum
    PerspectiveCameraFrustum(left=-1.0264004785593346, right=1.0264004785593346, bottom=-0.5773502691896257, top=0.5773502691896257, near=1.0, far=10000.0)
    """

    left: float = field(init=False)
    right: float = field(init=False)
    bottom: float = field(init=False)
    top: float = field(init=False)
    near: float = field(init=False)
    far: float = field(init=False)

    # These are the parameters used to create the frustum
    perspective_camera: InitVar[PerspectiveCamera]
    frame_width: InitVar[int]
    frame_height: InitVar[int]

    def __post_init__(
        self, perspective_camera: PerspectiveCamera, frame_width: int, frame_height: int
    ):
        """Create a perspective frustum from a perspective camera and frame dimensions.

        Parameters
        ----------
        perspective_camera : PerspectiveCamera
            The perspective camera.
        frame_width : int
            The width of the frame.
        frame_height : int
            The height of the frame.

        Examples
        --------
        >>> camera = PerspectiveCamera()
        >>> frustum = PerspectiveCameraFrustum(perspective_camera=camera, frame_width=1920, frame_height=1080)
        >>> frustum.left
        -1.0264004785593346
        >>> frustum.right
        1.0264004785593346
        >>> frustum.bottom
        -0.5773502691896257
        >>> frustum.top
        0.5773502691896257
        >>> frustum.near
        1.0
        >>> frustum.far
        10000.0
        """
        # Calculate the width and height at the near plane
        height_at_near = (
            2.0
            * perspective_camera.near
            * np.tan(0.5 * np.radians(perspective_camera.vertical_fov_degrees))
        )
        width_at_near = frame_width * height_at_near / frame_height

        self.left = -0.5 * width_at_near
        self.right = 0.5 * width_at_near
        self.bottom = -0.5 * height_at_near
        self.top = 0.5 * height_at_near
        self.near = perspective_camera.near
        self.far = perspective_camera.far


class OriginPointLocation(Enum):
    """The possible locations of the origin point in the frame.

    Notes
    -----
    From experience, the origin point is usually in the bottom left corner, but it can also be in the top left corner.
    The reason we need to know this is that the y-axis is inverted when the origin point is in the top left corner.
    """

    BOTTOM_LEFT_CORNER = auto()
    TOP_LEFT_CORNER = auto()


@dataclass(frozen=True)
class Environment:
    """The environment in which face geometry is calculated, defined by the origin point location and perspective camera.

    Attributes
    ----------
    origin_point_location : OriginPointLocation
        The location of the origin point in the frame.
    perspective_camera : PerspectiveCamera
        The perspective camera.

    Examples
    --------
    >>> Environment(OriginPointLocation.BOTTOM_LEFT_CORNER, PerspectiveCamera())
    Environment(origin_point_location=<OriginPointLocation.BOTTOM_LEFT_CORNER: 1>, perspective_camera=PerspectiveCamera(vertical_fov_degrees=60, near=1.0, far=10000.0))

    >>> Environment(OriginPointLocation.TOP_LEFT_CORNER, PerspectiveCamera(vertical_fov_degrees=45, near=0.1, far=100.0))
    Environment(origin_point_location=<OriginPointLocation.TOP_LEFT_CORNER: 2>, perspective_camera=PerspectiveCamera(vertical_fov_degrees=45, near=0.1, far=100.0))
    """

    origin_point_location: OriginPointLocation
    perspective_camera: PerspectiveCamera

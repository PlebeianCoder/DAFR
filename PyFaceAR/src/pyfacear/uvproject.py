from __future__ import annotations

from typing import Any
from typing import Optional
from typing import TypeVar

import cv2
import numpy as np

from .face_geometry import PerspectiveCamera
from .mesh import Mesh

N = TypeVar("N", int, np.int64, np.int16)

# We assume that the camera looks along the negative z-axis
view_direction = np.array([0, 0, -1], dtype=np.float32)


def compute_normal(triangle: np.ndarray[(3, 3), Any]) -> np.ndarray[(3,), Any]:
    """
    Compute the normal of the given triangle.

    Parameters
    ----------
    triangle: np.ndarray[(3, 3), Any]
        The triangle to compute the normal of.

    Returns
    -------
    np.ndarray[(3,), Any]
        The normal of the triangle.
    """
    v1 = triangle[1] - triangle[0]
    v2 = triangle[2] - triangle[0]
    return np.cross(v1, v2)


def calculate_barycentric(
    triangle: np.ndarray[(3, 2), Any], points: np.ndarray[(Any, 2), Any]
) -> np.ndarray[(Any, 3), Any]:
    """
    Calculate the barycentric coordinates of the given points in the given triangle.

    Parameters
    ----------
    triangle: np.ndarray[(3, 2), Any]
        The triangle to calculate the barycentric coordinates in.
    points: np.ndarray[(Any, 2), Any]
        The points to calculate the barycentric coordinates for.

    Returns
    -------
    np.ndarray[(Any, 3), Any]
        The barycentric coordinates of the points in the triangle.
    """
    v0, v1, v2 = triangle

    # Compute the area of the triangle
    area = np.abs(np.cross(v1 - v0, v2 - v0))

    if area == 0:
        return np.zeros((points.shape[0], 3))

    # Compute the areas of the sub-triangles
    area0 = np.abs(np.cross(points - v1, points - v2)) / area
    area1 = np.abs(np.cross(points - v2, points - v0)) / area
    area2 = 1 - area0 - area1

    return np.c_[area0, area1, area2]


def compute_depth(
    vertices: np.ndarray[(N, 2), Any],
    depths: np.ndarray[(N, 1), Any],
    indices: np.ndarray[(Any,), np.dtype[np.uint16]],
    viewport_width: int,
    viewport_height: int,
) -> np.ndarray[(Any, Any), np.dtype[np.float32]]:
    """
    Compute the depth buffer for the given vertices and indices.

    Parameters
    ----------
    vertices: np.ndarray[(N, 2), Any]
        The vertices to compute the depth buffer for.
    depths: np.ndarray[(N, 1), Any]
        The depths of the vertices.
    indices: np.ndarray[(Any,), np.dtype[np.uint16]]
        The indices of the vertices (into triangles).
    viewport_width: int
        The width of the viewport.
    viewport_height: int
        The height of the viewport.
    Returns
    -------
    np.ndarray[(Any, Any), np.dtype[np.float32]]
        The computed depth buffer.
    """
    # Create the depth buffer and fill it with negative infinity
    depth_buffer = np.zeros((viewport_height, viewport_width), dtype=np.float32)
    depth_buffer.fill(-np.inf)

    # Round the vertices to the nearest integer, so they are pixel aligned
    vertices = np.round(vertices).astype(np.int32)

    # Loop over each triangle
    for i in range(len(indices) // 3):
        # Get the triangle and the depths of each vertex
        triangle = vertices[indices[i * 3 : i * 3 + 3]]
        depth = depths[indices[i * 3 : i * 3 + 3]]

        # Compute the square region of the depth buffer that we will update
        min_x = np.min(triangle[:, 0]).clip(0, viewport_width)
        max_x = np.max(triangle[:, 0]).clip(0, viewport_width)
        min_y = np.min(triangle[:, 1]).clip(0, viewport_height)
        max_y = np.max(triangle[:, 1]).clip(0, viewport_height)

        # Get the coordinates of each point in the square region
        x, y = np.meshgrid(np.arange(min_x, max_x), np.arange(min_y, max_y))
        points = np.vstack((x.flatten(), y.flatten())).T

        # Calculate the barycentric coordinates of each point
        bary = calculate_barycentric(triangle, points)

        # We only want to keep the points that are inside the triangle
        bary_indices = np.where(np.all(bary >= 0, axis=1))

        # Get the pixel coordinates of the points inside the triangle
        px = points[bary_indices][:, 0]
        py = points[bary_indices][:, 1]

        # Use the barycentric coordinates to calculate the depth of each point
        bary_depth = bary[bary_indices].dot(depth)

        # Update the depth buffer to be the maximum of the current depth value and the new depth value
        depth_buffer[py, px] = np.maximum(depth_buffer[py, px], bary_depth)

    return depth_buffer


def homogenise(x: np.ndarray[(Any, Any), Any]) -> np.ndarray[(Any, Any), Any]:
    """
    Homogenise the given points.

    Parameters
    ----------
    x: np.ndarray[(Any, Any), Any]
        The points to homogenise.

    Returns
    -------
    np.ndarray[(Any, Any), Any]
        The homogenised points.
    """
    return np.hstack((x, np.ones((x.shape[0], 1))))


def project_and_uv_unwrap(
    matrix: np.ndarray[(Any, Any, Any), Any],
    camera: PerspectiveCamera,
    mesh: Mesh,
    model_matrix: np.ndarray[(4, 4), Any],
    result_width: int,
    result_height: int,
    back_face_culling: bool = True,
    occlusion_mesh: Optional[Mesh] = None,
) -> tuple[
    np.ndarray[(Any, Any, Any), Any], np.ndarray[(Any, Any), np.dtype[np.bool_]]
]:
    """
    Project the given matrix onto the given mesh and read the uv unwrapped matrix.

    Parameters
    ----------
    matrix: np.ndarray[(Any, Any, Any), Any]
        The matrix to project onto the mesh.
    camera: PerspectiveCamera
        The perspective camera used in rendering.
    mesh: Mesh
        The mesh to project the image onto.
    model_matrix: np.ndarray[(4, 4), np.float32]
        The model matrix to apply to the mesh.
    result_width: int
        The width of the resulting UV unwrapped matrix.
    result_height: int
        The height of the resulting UV unwrapped matrix.
    back_face_culling: bool
        Whether to perform back face culling (by default True).
    occlusion_mesh: Optional[Mesh]
        The occlusion mesh to occlude parts of the effect mesh.

    Notes
    -----
    The occlusion mesh is used to occlude parts of the given mesh and is particularly useful when trying to project
    a texture onto a face effect mesh. In this case, the occlusion mesh should be the face mesh from the face geometry.
    It is important to note that the occlusion mesh uses the same model matrix as the mesh to project onto but is placed
    slightly behind the mesh to project onto so that it doesn't exactly overlap and cause z-fighting.

    Returns
    -------
    tuple[np.ndarray[(Any, Any, Any), np.dtype[np.uint8]], np.ndarray[(Any, Any), np.dtype[np.bool_]]]
        The UV unwrapped matrix and the visible mask.
    """
    viewport_height, viewport_width = matrix.shape[:2]

    projection_matrix = camera.projection_matrix(viewport_width, viewport_height)

    # Apply the model matrix to the mesh
    mesh_vertices = mesh.vertices.copy()
    mesh_vertices = model_matrix @ homogenise(mesh_vertices).T

    # Get the depth values for the mesh
    mesh_depths = mesh_vertices[2]

    # Apply the projection matrix to the mesh
    mesh_vertices = projection_matrix @ mesh_vertices
    mesh_vertices = mesh_vertices[:3].T

    # If we use backface culling then we calculate the front facing triangles using the normals of each triangle
    mesh_triangles = mesh.triangles(mesh_vertices)

    if back_face_culling:
        mesh_normals = np.array(
            [compute_normal(triangle) for triangle in mesh_triangles]
        )

        front_facing_triangles = mesh_normals.dot(view_direction) < 0
    else:
        front_facing_triangles = np.ones(
            mesh.triangles(mesh_vertices).shape[0], dtype=bool
        )

    # We then divide by the z component to get the xy coordinates
    mesh_vertices = mesh_vertices / mesh_vertices[:, 2][:, np.newaxis]

    # We then scale the xy coordinates to the viewport
    mesh_vertices[:, 0] = (mesh_vertices[:, 0] + 1) * viewport_width / 2
    mesh_vertices[:, 1] = (mesh_vertices[:, 1] + 1) * viewport_height / 2

    # We then remove the z component
    mesh_vertices = mesh_vertices[:, :2]

    if occlusion_mesh is not None:
        # Calculate the occlusion transform matrix from the model matrix
        occlusion_transform_matrix = model_matrix.copy()
        occlusion_transform_matrix[2, 3] -= 0.2

        # Apply the pose transform matrix to the occlusion mesh
        occlusion_mesh_vertices = occlusion_mesh.vertices.copy()
        occlusion_mesh_vertices = (
            occlusion_transform_matrix @ homogenise(occlusion_mesh_vertices).T
        )

        # Get the depth values for the occlusion mesh
        occlusion_mesh_depths = occlusion_mesh_vertices[2]

        # Apply the projection matrix to the occlusion mesh
        occlusion_mesh_vertices = projection_matrix @ occlusion_mesh_vertices
        occlusion_mesh_vertices = occlusion_mesh_vertices[:3].T

        # Divide by the z component to get the xy coordinates
        occlusion_mesh_vertices = (
            occlusion_mesh_vertices / occlusion_mesh_vertices[:, 2][:, np.newaxis]
        )

        # Scale the xy coordinates to the viewport
        occlusion_mesh_vertices[:, 0] = (
            (occlusion_mesh_vertices[:, 0] + 1) * viewport_width / 2
        )
        occlusion_mesh_vertices[:, 1] = (
            (occlusion_mesh_vertices[:, 1] + 1) * viewport_height / 2
        )

        # Remove the z component
        occlusion_mesh_vertices = occlusion_mesh_vertices[:, :2]

        # Compute the depth buffer for the occlusion mesh
        occlusion_mesh_depth_buffer = compute_depth(
            occlusion_mesh_vertices,
            occlusion_mesh_depths,
            occlusion_mesh.indices,
            viewport_width,
            viewport_height,
        )

        # Write the mesh to the depth buffer
        mesh_depth_buffer = compute_depth(
            mesh_vertices, mesh_depths, mesh.indices, viewport_width, viewport_height
        )

        # Using the depth buffers, construct a mask for the occluded region
        visible_mask = mesh_depth_buffer > occlusion_mesh_depth_buffer

        # Bitwise AND the mask with the original depth buffer
        visible_mask = np.bitwise_and(mesh_depth_buffer > -np.inf, visible_mask)
    else:
        # Create the visible mask by filling in the area where the mesh is visible
        visible_mask = np.zeros((viewport_height, viewport_width), dtype=np.uint8)
        cv2.fillPoly(visible_mask, mesh.triangles(mesh_vertices).astype(np.int32), 1)
        visible_mask = visible_mask.astype(bool)

    # Mask the rendered image
    matrix = cv2.bitwise_and(matrix, matrix, mask=visible_mask.astype(np.uint8))

    # We then remove the vertices that are not visible
    mesh_triangles = mesh.triangles(mesh_vertices)[front_facing_triangles].astype(
        np.float32
    )

    # We then calculate the uv coordinates of the effect mesh and scale to the resulting image
    effect_mesh_uvs = mesh.texture_coords.copy()
    effect_mesh_uvs[:, 0] = effect_mesh_uvs[:, 0] * result_width
    effect_mesh_uvs[:, 1] = effect_mesh_uvs[:, 1] * result_height

    # Convert the uvs into triangles
    effect_mesh_uv_triangles = mesh.triangles(effect_mesh_uvs)[front_facing_triangles]

    # We then create the result matrix which has the height and width as given with the number of channels as the image
    result = np.zeros(
        (result_height, result_width, matrix.shape[2]), dtype=matrix.dtype
    )

    mask = np.zeros((result_height, result_width), dtype=np.bool_)

    # We can then use the uv coordinates and the triangles to reverse UV unwrap the image
    for i in range(effect_mesh_uv_triangles.shape[0]):
        # We need to find the corresponding uv coordinates
        uv_triangle = effect_mesh_uv_triangles[i]
        # We need to find the corresponding xy coordinates
        xy_triangle = mesh_triangles[i]

        # We then perform an affine transformation from the xy coordinates to the uv coordinates
        transform = cv2.getAffineTransform(xy_triangle, uv_triangle)
        output = cv2.warpAffine(matrix, transform, (result_width, result_height))

        # Mask out where the triangle is in the uv image
        sub_mask = np.zeros((result_height, result_width), dtype=matrix.dtype)
        cv2.fillPoly(sub_mask, [uv_triangle.astype(np.int32)], 1)

        # Add the sub mask to the mask
        mask = mask | sub_mask.astype(np.bool_)

        # Use the mask to add the output to the result
        result = (
            result * (1 - sub_mask[:, :, np.newaxis])
            + output * sub_mask[:, :, np.newaxis]
        )

    return result, mask

from __future__ import annotations

import json
from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import TypeVar

import numpy as np
from OpenGL.GL import *
from pyfacear.data import read_text
from pyfacear.utils import Initialisable
from pyfacear.utils import require_initialised
from pyfacear.utils import singleton


M = TypeVar("M", int, np.int64, np.int16)


@dataclass
class Mesh(Initialisable):
    """
    A class to represent a mesh in OpenGL.
    """

    # The vertices of a mesh represented by a NxM array of floats where N is the number of vertices and M is the number
    # of components per vertex (e.g. 3 for x, y, z)
    vertices: np.ndarray[(Any, Any), np.dtype[np.float32]]
    # The texture coordinates of a mesh represented by a Nx2 array of floats where N is the number of vertices
    texture_coords: np.ndarray[(Any, 2), np.dtype[np.float32]]
    # The indices of a mesh is represented by a 1D array
    indices: np.ndarray[(Any,), np.dtype[np.uint16]]
    # The primitive type of the mesh (GL_TRIANGLES, GL_LINES, etc.)
    primitive_type: GLint = GL_TRIANGLES

    # The following are private fields that are used to store the vertex array object (VAO) and the vertex buffer etc.
    __vao: GLuint = field(init=False, default=0, repr=False)
    __indices_vbo: GLuint = field(init=False, default=0, repr=False)
    __vertex_vbo: GLuint = field(init=False, default=0, repr=False)
    __texture_coords_vbo: GLuint = field(init=False, default=0, repr=False)

    def __post_init__(self):
        # After initialisation, we must call the super class's __init__ method which setups the initialisation flag
        super().__init__()

    @property
    def vertex_count(self) -> int:
        """
        :return: The number of vertices in the mesh.
        """
        return self.vertices.shape[0]

    @property
    def vertex_size(self) -> int:
        """
        :return: The number of components per vertex.
        """
        return self.vertices.shape[1]

    @property
    def texture_coordinate_size(self) -> int:
        """
        :return: The number of components per texture coordinate.
        """
        return self.texture_coords.shape[1]

    @property
    def vao(self) -> GLuint:
        """
        :return: The vertex array object (VAO) of the mesh.
        """
        return self.__vao

    def triangles(
        self, positions: np.ndarray[(Any, M), Any]
    ) -> np.ndarray[(Any, 3, M), Any]:
        """
        Convert the positions to triangles using the indices of the mesh.

        Parameters
        ----------
        positions: np.ndarray[(Any, M), Any]
            The positions to convert to triangles.

        Returns
        -------
        np.ndarray[(Any, 3, M), Any]
            The positions as triangles.
        """
        return positions[self.indices].reshape(-1, 3, positions.shape[1])

    @property
    def vertex_triangles(self) -> np.ndarray[(Any, 3, 3), np.dtype[np.float32]]:
        """
        Convert the vertices to triangles using the indices of the mesh.

        Returns
        -------
        np.ndarray[(Any, 3, 3), np.dtype[np.float32]]
            The vertices as triangles.
        """
        return self.triangles(self.vertices)

    @property
    def texture_coords_triangles(self) -> np.ndarray[(Any, 3, 2), np.dtype[np.float32]]:
        """
        Convert the texture coordinates to triangles using the indices of the mesh.

        Returns
        -------
        np.ndarray[(Any, 3, 2), np.dtype[np.float32]]
            The texture coordinates as triangles.
        """
        return self.triangles(self.texture_coords)

    def init(self):
        """
        Initialises the mesh by creating the vertex array object (VAO) and the vertex buffer objects (VBOs) for the
        vertices, texture coordinates and indices.
        """
        # Generate the vertex array object (VAO) and bind it
        self.__vao = glGenVertexArrays(1)
        glBindVertexArray(self.__vao)

        # Generate the indices VBO and bind it
        self.__indices_vbo = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.__indices_vbo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, self.indices, GL_STATIC_DRAW)

        # Generate the vertex VBO and bind it
        self.__vertex_vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.__vertex_vbo)
        glBufferData(GL_ARRAY_BUFFER, self.vertices.ravel(), GL_STATIC_DRAW)

        # Set the vertex attribute pointer for the vertices (location 0)
        glVertexAttribPointer(0, self.vertex_size, GL_FLOAT, GL_FALSE, 0, None)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

        # Generate the texture coordinates VBO and bind it
        self.__texture_coords_vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.__texture_coords_vbo)
        glBufferData(GL_ARRAY_BUFFER, self.texture_coords.ravel(), GL_STATIC_DRAW)

        # Set the vertex attribute pointer for the texture coordinates (location 1)
        glVertexAttribPointer(
            1, self.texture_coordinate_size, GL_FLOAT, GL_FALSE, 0, None
        )
        glBindBuffer(GL_ARRAY_BUFFER, 0)

        # Unbind the VAO
        glBindVertexArray(0)

        # Call the super class's init method to set the initialisation flag
        super().init()

    def exit(self):
        """
        Clean up the VAO and VBOs.
        """
        # Call the super class's exit method to reset the initialisation flag
        super().exit()
        # Delete the VAO and VBOs
        if self.__vao:
            glDeleteVertexArrays(1, [self.__vao])
            self.__vao = 0
        if self.__indices_vbo:
            glDeleteBuffers(1, [self.__indices_vbo])
            self.__indices_vbo = 0
        if self.__vertex_vbo:
            glDeleteBuffers(1, [self.__vertex_vbo])
            self.__vertex_vbo = 0
        if self.__texture_coords_vbo:
            glDeleteBuffers(1, [self.__texture_coords_vbo])
            self.__texture_coords_vbo = 0

    @require_initialised
    def enable(self):
        """
        Enable the VAO and vertex attribute arrays.
        """
        glBindVertexArray(self.__vao)
        glEnableVertexAttribArray(0)
        glEnableVertexAttribArray(1)

    @require_initialised
    def disable(self):
        """
        Disable the vertex attribute arrays and unbind the VAO
        """
        glDisableVertexAttribArray(0)
        glDisableVertexAttribArray(1)
        glBindVertexArray(0)

    @staticmethod
    def create_quad_mesh() -> "Mesh":
        """
        Create a quad mesh.

        :return: A mesh representing a quad.
        """
        return Mesh(
            vertices=np.array(
                [
                    [-1, 1, 0],
                    [-1, -1, 0],
                    [1, -1, 0],
                    [1, 1, 0],
                ],
                dtype=np.float32,
            ),
            texture_coords=np.array(
                [
                    [0, 1],
                    [0, 0],
                    [1, 0],
                    [1, 1],
                ],
                dtype=np.float32,
            ),
            indices=np.array([0, 1, 3, 3, 1, 2], dtype=np.uint16),
            primitive_type=GL_TRIANGLES,
        )

    def flip_texture_coords(self):
        """
        Flip the texture coordinates v coordinate.
        """
        self.texture_coords[:, 1] = 1 - self.texture_coords[:, 1]


@dataclass
class CanonicalFaceMesh(Mesh):
    """
    A class to represent the canonical face mesh.
    """

    weights: np.ndarray[(Any, 1), np.dtype[np.float64]] = None

    def __post_init__(self):
        super().__post_init__()

    @staticmethod
    @singleton
    def get() -> "CanonicalFaceMesh":
        """
        Get the canonical face mesh.

        :return: The canonical face mesh.
        """
        from pyfacear.mesh import OBJMeshIO

        obj = read_text("canonical_face_model.obj")
        mesh = OBJMeshIO.load_raw(obj)

        basis = json.loads(read_text("landmark_basis.json"))

        weights = np.zeros((mesh.vertex_count, 1), dtype=np.float64)
        for landmark_id, weight in basis.items():
            weights[int(landmark_id)] = weight

        # Flip the z-coordinate of the canonical mesh so the landmarks are applied correctly
        mesh.vertices[:, 2] *= -1

        return CanonicalFaceMesh(
            vertices=mesh.vertices,
            texture_coords=mesh.texture_coords,
            indices=mesh.indices,
            weights=weights,
        )

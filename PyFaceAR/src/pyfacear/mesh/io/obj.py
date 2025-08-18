import numpy as np
from OpenGL.GL import GL_TRIANGLES
from pyfacear.mesh.base import Mesh
from pyfacear.mesh.io.base import MeshIO


class OBJMeshIO(MeshIO):
    @staticmethod
    def load_raw(data: str) -> Mesh:
        """
        Create a mesh from an OBJ file.

        Parameters
        ----------
        data : str
            The OBJ contents.

        Returns
        -------
        Mesh
            The OpenGL mesh representation of the loaded OBJ model.
        """
        # Split the obj file into lines
        lines = data.splitlines()

        vertices = []
        texture_coords = []
        indices = []
        texture_indices = {}

        # Parse the obj file
        for line in lines:
            # Parse the vertex
            if line.startswith("v "):
                vertex = list(map(float, line.split()[1:]))
                vertices.append(vertex)
            elif line.startswith("vt "):
                # Parse the texture coordinate
                t = list(map(float, line.split()[1:]))
                texture_coords.append(t)
            elif line.startswith("f "):
                # Parse the face
                face = line.split()[1:]
                # For each vertex in the face, the first index is the vertex index and the second index is the texture
                # coordinate index
                for vertex in face:
                    vertex = vertex.split("/")
                    vertex_index = int(vertex[0]) - 1
                    indices.append(vertex_index)
                    texture_indices[vertex_index] = int(vertex[1]) - 1

        # Re-order the texture coordinates according to the indices
        texture_coords = [
            texture_coords[texture_indices[i]] for i in range(len(vertices))
        ]
        # Flip the texture coordinates v coordinate
        texture_coords = [[t[0], 1 - t[1]] for t in texture_coords]

        return Mesh(
            vertices=np.array(vertices, dtype=np.float32),
            texture_coords=np.array(texture_coords, dtype=np.float32),
            indices=np.array(indices, dtype=np.uint16),
            primitive_type=GL_TRIANGLES,
        )

    @staticmethod
    def export_raw(mesh: Mesh) -> str:
        obj = ""
        for vertex in mesh.vertices:
            obj += f"v {vertex[0]} {vertex[1]} {vertex[2]}\n"
        # Note: we flip the texture coordinate when saving
        for texture_coord in mesh.texture_coords:
            obj += f"vt {texture_coord[0]} {1 - texture_coord[1]}\n"
        for i in range(0, len(mesh.indices), 3):
            obj += (
                f"f {mesh.indices[i] + 1}/{mesh.indices[i] + 1} {mesh.indices[i + 1] + 1}/{mesh.indices[i + 1] + 1}"
                f" {mesh.indices[i + 2] + 1}/{mesh.indices[i + 2] + 1}\n"
            )
        return obj

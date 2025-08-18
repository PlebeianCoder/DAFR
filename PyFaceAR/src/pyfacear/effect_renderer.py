from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import Optional

import numpy as np

from .face_geometry import Environment
from .face_geometry import FaceGeometry
from .mesh import Mesh
from .render_target import RenderTarget
from .renderer import Renderer
from .renderer import RenderMode
from .texture import Texture
from .utils import Initialisable
from .utils import require_initialised


identity_matrix = np.identity(4, dtype=np.float32)


@dataclass
class EffectRenderer(Initialisable):
    """
    A class which can render a 3D effect to a texture using a face geometry.
    """

    environment: Environment
    effect_texture: Texture
    effect_mesh: Optional[Mesh] = None

    __render_target: RenderTarget = field(init=False, repr=False)
    __renderer: Renderer = field(init=False, repr=False)
    __renderable_quad_mesh: Mesh = field(init=False, repr=False)
    __empty_colour_texture: Texture = field(init=False, repr=False)

    def __post_init__(self):
        # After initialisation, we must call the super class's __init__ method which setups the initialisation flag
        super().__init__()
        # Create the render target, renderer, quad mesh and empty colour texture
        self.__render_target = RenderTarget()
        self.__renderer = Renderer()
        self.__renderable_quad_mesh = Mesh.create_quad_mesh()
        self.__empty_colour_texture = Texture.create_empty()

    def init(self):
        """
        Initialise the effect renderer, initialising the render target, renderer, quad mesh and empty colour texture.
        """
        self.__render_target.init()
        self.__renderer.init()
        self.__renderable_quad_mesh.init()
        self.__empty_colour_texture.init()
        self.effect_texture.init()
        if self.effect_mesh is not None:
            self.effect_mesh.init()

        # Call the super class's init method to set the initialisation flag
        super().init()

    def exit(self):
        """
        Clean up the effect renderer, deleting the render target, renderer, quad mesh and empty colour texture.
        """
        # Call the super class's exit method to reset the initialisation flag
        super().exit()

        self.__render_target.exit()
        self.__renderer.exit()
        self.__renderable_quad_mesh.exit()
        self.__empty_colour_texture.exit()
        self.effect_texture.exit()
        if self.effect_mesh is not None:
            self.effect_mesh.exit()

    @require_initialised
    def render_effect(
        self,
        face_geometries: list[FaceGeometry],
        image_width: int,
        image_height: int,
        source_texture: Texture,
        destination_texture: Texture,
    ):
        """
        Render the effect to the destination texture using the source texture and the face geometries.

        :param face_geometries: The face geometries to render the effect to.
        :param image_width: The width of the image.
        :param image_height: The height of the image.
        :param source_texture: The source texture to render the effect on top of.
        :param destination_texture: The destination texture to render the effect to.
        """
        # Set the destination texture as the colour buffer. Then, clear both the colour and the
        # depth buffers for the render target
        self.__render_target.set_colour_buffer(destination_texture)
        self.__render_target.clear()

        # Render the source texture on top of the quad mesh (i.e. make a copy) into the render target
        self.__renderer.render(
            self.__render_target,
            source_texture,
            self.__renderable_quad_mesh,
            identity_matrix,
            identity_matrix,
            RenderMode.OVERDRAW,
        )

        perspective_matrix = self.environment.perspective_camera.projection_matrix(
            image_width, image_height
        )

        # Initialise all the meshes
        for face_geometry in face_geometries:
            face_geometry.mesh.init()

        # Render the face mesh to occlude the effect
        for face_geometry in face_geometries:
            # Render the face mesh using the empty colour texture, i.e. the face mesh occluder
            # For occlusion, the pose transformation is moved away from camera in order to allow the face
            # mesh texture to be rendered without failing the depth test
            occlusion_face_pose_transform_matrix = (
                face_geometry.pose_transform_matrix.copy()
            )
            occlusion_face_pose_transform_matrix[2, 3] -= 0.2
            self.__renderer.render(
                self.__render_target,
                self.__empty_colour_texture,
                face_geometry.mesh,
                perspective_matrix,
                occlusion_face_pose_transform_matrix,
                RenderMode.OCCLUSION,
            )

        # Render the main face mesh effect component for each face
        for face_geometry in face_geometries:
            # If there is no effect 3D mesh provided, then the face mesh itself is used as
            # a topology for rendering (for example, this can be used for facepaint effects or
            # AR makeup)
            main_mesh_effect = (
                self.effect_mesh if self.effect_mesh is not None else face_geometry.mesh
            )
            self.__renderer.render(
                self.__render_target,
                self.effect_texture,
                main_mesh_effect,
                perspective_matrix,
                face_geometry.pose_transform_matrix,
                RenderMode.OPAQUE,
            )

        # Clean up all the meshes
        for face_geometry in face_geometries:
            face_geometry.mesh.exit()

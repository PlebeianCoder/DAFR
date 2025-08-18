from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from enum import auto
from enum import Enum

import numpy as np
from OpenGL.GL import *

from .mesh import Mesh
from .render_target import RenderTarget
from .shader import Shader
from .shader import ShaderProgram
from .texture import Texture
from .utils import Initialisable
from .utils import require_initialised


class RenderMode(Enum):
    """
    An enumeration to represent the different render modes.
    """

    # Opaque draws with blending and depth testing enabled
    OPAQUE = auto()
    # Overdraw draws only to the colour buffer with depth testing disabled
    OVERDRAW = auto()
    # Occlusion draws only to the depth buffer with depth testing enabled
    OCCLUSION = auto()


class Attributes(Enum):
    """
    An enumeration to represent the different vertex attributes for the default shader used by the renderer.
    """

    VERTEX = 0
    TEXTURE = 1


class Uniforms(Enum):
    """
    An enumeration to represent the different uniforms for the default shader used by the renderer.
    """

    PROJECTION_MAT = "projection_mat"
    MODEL_MAT = "model_mat"
    TEXTURE = "texture"


# The default vertex and fragment shaders
default_vertex_shader = """
uniform mat4 projection_mat;
uniform mat4 model_mat;

attribute vec4 position;
attribute vec2 tex_coord;

varying vec2 v_tex_coord;

void main() {
    v_tex_coord = tex_coord;
    gl_Position = projection_mat * model_mat * position;
}
"""

default_fragment_shader = """
varying vec2 v_tex_coord;
uniform sampler2D texture;

void main() {
    gl_FragColor = texture2D(texture, v_tex_coord);
}
"""

vertex_shader = Shader(default_vertex_shader, GL_VERTEX_SHADER)
fragment_shader = Shader(default_fragment_shader, GL_FRAGMENT_SHADER)


@dataclass
class Renderer(Initialisable):
    """
    A class which contains a default shader to render a mesh with a texture.
    """

    __shader_program: ShaderProgram = field(
        default_factory=lambda: ShaderProgram(
            [vertex_shader, fragment_shader], Attributes, Uniforms
        ),
        repr=False,
        init=False,
    )

    def __post_init__(self):
        # After initialisation, we must call the super class's __init__ method which setups the initialisation flag
        super().__init__()

    def init(self):
        """
        Initialise the renderer, initialising the default shader program.
        """
        # Initialise the shader program
        self.__shader_program.init()

        # Call the super class's init method to set the initialisation flag
        super().init()

    def exit(self):
        """
        Clean up the shader, deleting the shader program.
        """
        # Call the super class's exit method to reset the initialisation flag
        super().exit()

        # Clean up the shader program
        self.__shader_program.exit()

    @require_initialised
    def render(
        self,
        render_target: RenderTarget,
        texture: Texture,
        mesh: Mesh,
        projection_matrix: np.ndarray[(4, 4), np.dtype[np.float32]],
        model_matrix: np.ndarray[(4, 4), np.dtype[np.float32]],
        render_mode: RenderMode,
    ):
        """
        Render the mesh with the texture to the render target using the default shader program.
        :param render_target: The texture to render to
        :param texture: The texture to apply to the mesh
        :param mesh: The mesh to render
        :param projection_matrix: The projection matrix
        :param model_matrix: The model matrix
        :param render_mode: The rendering mode to use. See RenderMode for more information.
        """
        self.__shader_program.use()

        # Set up the GL state
        glEnable(GL_BLEND)
        glFrontFace(GL_CCW)

        if render_mode == RenderMode.OPAQUE:
            # In opaque mode, we use alpha blending and depth testing (with depth writing enabled)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            glEnable(GL_DEPTH_TEST)
            glDepthMask(GL_TRUE)
        elif render_mode == RenderMode.OVERDRAW:
            # In overdraw mode, we disable depth testing (with depth writing disabled)
            glBlendFunc(GL_ONE, GL_ZERO)
            glDisable(GL_DEPTH_TEST)
            glDepthMask(GL_FALSE)
        elif render_mode == RenderMode.OCCLUSION:
            # In occlusion mode, we only write to the depth buffer (with depth testing enabled)
            glBlendFunc(GL_ZERO, GL_ONE)
            glEnable(GL_DEPTH_TEST)
            glDepthMask(GL_TRUE)

        render_target.bind()

        # Set up vertex attributes
        mesh.enable()

        # Set up textures and uniforms
        glActiveTexture(GL_TEXTURE1)
        texture.bind()
        self.__shader_program.load_uniform(Uniforms.PROJECTION_MAT, projection_matrix)
        self.__shader_program.load_uniform(Uniforms.MODEL_MAT, model_matrix)
        self.__shader_program.load_uniform(Uniforms.TEXTURE, 1)

        # Draw the mesh
        glDrawElements(mesh.primitive_type, mesh.indices.size, GL_UNSIGNED_SHORT, None)

        # Unbind textures and uniforms
        glActiveTexture(GL_TEXTURE1)
        texture.unbind()
        mesh.disable()
        render_target.unbind()

        # Restore the GL state
        glDepthMask(GL_FALSE)
        glDisable(GL_DEPTH_TEST)
        glDisable(GL_BLEND)

        self.__shader_program.stop()
        glFlush()

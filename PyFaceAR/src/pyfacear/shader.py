from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from enum import Enum
from typing import TypeVar
from typing import Union

import numpy as np
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram
from OpenGL.GL.shaders import compileShader

from .utils import Initialisable
from .utils import require_initialised

T = TypeVar("T", str, Enum)


# noinspection PyShadowingNames
def _get_name(name: T) -> str:
    """
    Get the name of the attribute or uniform.
    :param name: The name of the attribute or uniform.

    :return: The name of the attribute or uniform.
    """
    if isinstance(name, Enum):
        if isinstance(name.value, str):
            return name.value
        else:
            return name.name.lower()
    return name


def _get_attribute_index(i: int, attribute: Union[str, Enum]) -> int:
    """
    Get the index of the attribute.
    :param i: The assumed index of the attribute.
    :param attribute: The attribute to get the index of.

    :return: The index of the attribute.
    """
    # If the attribute is an enum and has an integer value, return the value
    if isinstance(attribute, Enum) and isinstance(attribute.value, int):
        return attribute.value
    return i


@dataclass
class Shader:
    """
    A class to represent a shader in OpenGL.
    """

    code: str
    type: GLenum


@dataclass
class ShaderProgram(Initialisable):
    """
    A class to represent a shader program in OpenGL.
    """

    shaders: list[Shader]
    attributes: Union[list[Union[str, Enum]], type[Enum]]
    uniforms: Union[list[T], type[Enum]]
    __uniform_locations: dict[T, GLint] = field(
        init=False, default_factory=dict, repr=False
    )
    __program: GLuint = field(init=False, default=0, repr=False)

    def __post_init__(self):
        # After initialisation, we must call the super class's __init__ method which setups the initialisation flag
        super().__init__()

    def init(self):
        """
        Initialise the shader program.
        """
        # Compile the shaders and link them to the program
        self.__program = compileProgram(
            *[compileShader(shader.code, shader.type) for shader in self.shaders]
        )

        # Bind the attributes to the program
        for i, attribute in enumerate(self.attributes):
            glBindAttribLocation(
                self.__program, _get_attribute_index(i, attribute), _get_name(attribute)
            )

        # Load the uniform locations
        for uniform in self.uniforms:
            self.__uniform_locations[uniform] = glGetUniformLocation(
                self.__program, _get_name(uniform)
            )

        # Call the super class's init method to set the initialisation flag
        super().init()

    def exit(self):
        """
        Cleanup the shader program.
        """
        # Call the super class's exit method to reset the initialisation flag
        super().exit()

        # If we have a program, delete it
        if self.__program:
            glDeleteProgram(self.__program)
            self.__program = 0
            self.__uniform_locations.clear()

    @require_initialised
    def use(self):
        """
        Use the shader program.
        """
        glUseProgram(self.__program)

    @require_initialised
    def stop(self):
        """
        Stop using the shader program.
        """
        glUseProgram(0)

    @require_initialised
    def get_uniform_location(self, uniform: T) -> GLint:
        """
        Get the location of a uniform.

        :param uniform: The uniform to get the location of.

        :return: The location of the uniform.
        """
        return self.__uniform_locations[uniform]

    @require_initialised
    def load_uniform(self, uniform: T, value: Union[int, float, np.ndarray]):
        """
        Load a uniform with a value.

        :param uniform: The uniform to load.
        :param value: The value to load into the uniform.
        """
        location = self.get_uniform_location(uniform)
        if isinstance(value, int):
            glUniform1i(location, value)
        elif isinstance(value, float):
            glUniform1f(location, value)
        elif isinstance(value, np.ndarray):
            if value.size == 1:
                glUniform1fv(location, 1, value)
            elif value.size == 2:
                glUniform2fv(location, 1, value)
            elif value.size == 3:
                glUniform3fv(location, 1, value)
            elif value.size == 4:
                glUniform4fv(location, 1, value)
            elif value.shape == (4, 4):
                # Numpy uses row-major order, so we need to transpose the matrix as OpenGL expects column-major order
                glUniformMatrix4fv(location, 1, GL_FALSE, value.T)
            else:
                raise ValueError(
                    "Unsupported value size for load_uniform. You may have to use the appropriate glUniform function."
                )
        else:
            raise ValueError(
                "Unsupported value type for load_uniform. You may have to use the appropriate glUniform function."
            )

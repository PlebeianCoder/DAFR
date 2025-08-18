"""An implementation of a texture that can represent an image and can be used in OpenGL.

This module contains a Texture implementation that can be used to store image data and can be bound to a texture target
in OpenGL. The texture can be updated with new image data and read back from the GPU.
"""
from __future__ import annotations

from typing import Any
from typing import Optional
from typing import Union

import numpy as np
from mediapipe import Image as MPImage
from OpenGL.GL import *
from PIL.Image import Image as PILImage

from .utils import Initialisable
from .utils import require_initialised


class Texture(Initialisable):
    # noinspection PyUnresolvedReferences
    """Image that can be used when rendering geometry in OpenGL.

    Image that can be used as a texture in OpenGL. That is, it can be bound to a texture target and used when rendering
    geometry. The image can be updated with new data and read back from the GPU.

    The image can be provided as a numpy array, a PIL image or a MediaPipe image. If the image is not provided, then an
    empty texture can be created with a given size.

    Attributes
    ----------
    image : np.ndarray[(Any, Any, Any), np.dtype[np.uint8]]
        The image data stored as width x height x 3 (RGB) or 4 (RGBA) numpy array.
    width : int
        The width of the image.
    height : int
        The height of the image.
    image_format : GLenum
        The OpenGL format of the image.
    target : GLenum
        The OpenGL target of the texture (GL_TEXTURE_2D).
    handle : GLint
        The OpenGL handle of the texture.

    Notes
    -----
    Before a texture can be used for rendering, it must first be initialised. This is done by calling the `init` method.
    Then, the texture can be bound to the texture target using the `bind` method. After the texture has been used, it
    should be unbound from the texture target using the `unbind` method. Finally, the texture can be de-initialised using
    the `exit` method.

    If a texture is to be used in many draw calls, then it is best to only call `exit` at the very end of the program.

    Examples
    --------
    Creating and using a texture made from a numpy array:

    >>> from pyfacear import OpenGLContext
    >>> import numpy as np
    >>> image = np.zeros((100, 100, 3), dtype=np.uint8)
    >>> context = OpenGLContext(100, 100) # Create an OpenGL context so we can initialise the texture
    >>> texture = Texture(image)
    >>> texture.init()
    >>> texture.bind()
    >>> # Render geometry
    >>> texture.unbind()
    >>> texture.exit()
    >>> context.exit()
    """

    def __init__(
        self,
        image: Optional[
            Union[
                np.ndarray[(Any, Any, Any), np.dtype[np.uint8]],
                PILImage,
                MPImage,
            ]
        ] = None,
        size: Optional[tuple[int, int]] = None,
        alpha: bool = False,
    ):
        """Create a texture from an image or create an empty texture.

        Parameters
        ----------
        image : numpy array of shape (height, width, channels) or PIL image or MediaPipe image, optional
            The image to create the texture from. If not defined, then an empty texture will be created.
        size : tuple[int, int], optional
            The size of the image (if defining an empty texture).
        alpha : bool, optional
            Whether to include alpha in the empty texture. By default, this is False.

        Raises
        ------
        ValueError
            If both image and size are defined or if neither image nor size are defined or
            if the provided image is not a numpy array, PIL image or MediaPipe image.

        Notes
        -----
        If the image is not provided, then an empty texture can be created with a given size.

        The given image is assumed to be in RGB or RGBA format. If the image is not in this format, then it must be
        converted before being passed to this class.

        Examples
        --------
        Creating a texture from a numpy array:

        >>> import numpy as np
        >>> image = np.zeros((100, 100, 3), dtype=np.uint8)
        >>> Texture(image)
        Texture(width=100, height=100, format=RGB)

        Creating a texture from a PIL image:

        >>> from PIL import Image
        >>> image = Image.new("RGBA", (100, 100))
        >>> Texture(image)
        Texture(width=100, height=100, format=RGBA)

        Creating a texture from a MediaPipe image:

        >>> from mediapipe import Image as MPImage
        >>> image = MPImage(100, 100)
        >>> Texture(image)
        Texture(width=100, height=100, format=RGB)

        Creating an empty texture:

        >>> Texture(size=(100, 100))
        Texture(width=100, height=100, format=RGB)

        Creating an empty texture with alpha:

        >>> Texture(size=(100, 100), alpha=True)
        Texture(width=100, height=100, format=RGBA)
        """
        # After initialisation, we must call the super class's __init__ method which setups the initialisation flag
        super().__init__()

        # If the image and size are both None or both not None, then raise an error
        if (image is not None) == (size is not None):
            raise ValueError("Either image or size must be provided, but not both")

        # If the image is None, then create an empty image
        if image is None:
            assert size is not None
            image = np.zeros((size[1], size[0], 3 if not alpha else 4), dtype=np.uint8)

        # The image must be a numpy array in RGB
        if isinstance(image, np.ndarray):
            self.__image = image.copy()
        elif isinstance(image, PILImage):
            self.__image = np.array(image)
        elif isinstance(image, MPImage):
            self.__image = image.numpy_view()
        else:
            raise ValueError("Invalid image type")

        self.__image_pointer = self.__image.ctypes.data_as(
            ctypes.POINTER(ctypes.c_uint8)
        )
        self.__texture_handle = 0

    @staticmethod
    def create_empty(size: tuple[int, int] = (1, 1), alpha: bool = False) -> "Texture":
        """Create an empty texture.

        Create an empty texture with the given size. The texture can be used to store image data, for example the
        output of a render target.

        Parameters
        ----------
        size : tuple[int, int], optional
            The size of the empty texture. By default, this is (1, 1).
        alpha : bool, optional
            Whether to include alpha in the empty texture. By default, this is False.

        Returns
        -------
        Texture
            The empty texture with the given size and alpha channel if specified.

        Examples
        --------
        Creating an empty texture:

        >>> Texture.create_empty((100, 100))
        Texture(width=100, height=100, format=RGB)

        Creating an empty texture with alpha:

        >>> Texture.create_empty((100, 100), alpha=True)
        Texture(width=100, height=100, format=RGBA)

        Notes
        -----
        The texture is not initialised and must be initialised before it can be used.

        """
        return Texture(size=size, alpha=alpha)

    @property
    def image(self) -> np.ndarray[(Any, Any, Any), np.dtype[np.uint8]]:
        """np.ndarray[(Any, Any, Any), np.dtype[np.uint8]]: The image data stored as width x height x 3 (RGB) or 4 (RGBA) numpy array.

        The image data can be updated by setting a new value to the image property. The new value must be either a
        numpy array, a PIL image or a MediaPipe image. This will update the texture with the new image data using the
        `update` method.
        """
        return self.__image

    @image.setter
    def image(
        self,
        value: Union[
            np.ndarray[(Any, Any, Any), np.dtype[np.uint8]],
            PILImage,
            MPImage,
        ],
    ):
        """Update the image data.

        Update the image data by providing a new image. The new image must be either a numpy array, a PIL image or a
        MediaPipe image. This will update the texture with the new image data using the `update` method.

        Parameters
        ----------
        value : numpy array of shape (height, width, channels) or PIL image or MediaPipe image
            The new image to update the texture with.
        """
        self.update(value)

    @property
    def width(self) -> int:
        """int: The width of the image."""
        return self.__image.shape[1]

    @property
    def height(self) -> int:
        """int: The height of the image."""
        return self.__image.shape[0]

    @property
    def image_format(self) -> GLenum:
        """GLenum: The OpenGL format of the image."""
        return GL_RGB if self.__image.shape[2] == 3 else GL_RGBA

    @property
    def target(self) -> GLenum:
        """GLenum: The OpenGL target of the texture (GL_TEXTURE_2D)."""
        return GL_TEXTURE_2D

    @property
    def handle(self) -> GLint:
        """GLint: The OpenGL texture handle."""
        return self.__texture_handle

    def __repr__(self) -> str:
        """str: Return a string representation of the texture."""
        return f"Texture(width={self.width}, height={self.height}, format={'RGB' if self.image_format == GL_RGB else 'RGBA'})"

    def init(self):
        """Initialise the texture and load the image data into the GPU."""
        # If the glGenTextures function is not available, then return
        if not bool(glGenTextures):
            return

        # Generate a texture handle
        self.__texture_handle = glGenTextures(1)
        # If the texture handle is invalid, then return
        if not self.__texture_handle:
            return

        # Bind the texture
        glBindTexture(self.target, self.handle)

        # Load the image data into the texture
        glTexImage2D(
            self.target,
            0,
            self.image_format,
            self.width,
            self.height,
            0,
            self.image_format,
            GL_UNSIGNED_BYTE,
            self.__image_pointer,
        )
        # Set the texture parameters
        glTexParameteri(self.target, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(self.target, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(self.target, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(self.target, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)

        # Unbind the texture
        glBindTexture(self.target, 0)

        # Call the super class's init method to set the initialisation flag
        super().init()

    def exit(self):
        """Cleanup the texture."""
        # Call the super class's exit method to reset the initialisation flag
        super().exit()

        # If the texture handle is valid, then delete the texture
        if self.__texture_handle:
            glDeleteTextures([self.__texture_handle])
            self.__texture_handle = 0

    @require_initialised
    def bind(self):
        """Bind the texture to the OpenGL target.

        Bind the texture to the OpenGL target, allowing the texture to be used when rendering geometry.

        Notes
        -----
        The texture must be initialised before it can be bound.

        Raises
        ------
        RuntimeError
            If the texture has not been initialised when the method is called.
        """
        glBindTexture(self.target, self.handle)

        if self.__image.shape[2] == 3:
            glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        else:
            glPixelStorei(GL_UNPACK_ALIGNMENT, 4)

    @require_initialised
    def unbind(self):
        """Unbind the texture from the OpenGL target.

        Unbind the texture from the OpenGL target, preventing the texture from being used when rendering geometry.

        Notes
        -----
        The texture must be initialised before it can be unbound.

        Raises
        ------
        RuntimeError
            If the texture has not been initialised when the method is called.
        """
        glBindTexture(self.target, 0)

    @require_initialised
    def read(
        self, shape: Optional[tuple[int, ...]] = None
    ) -> np.ndarray[Any, np.dtype[np.uint8]]:
        """Read the image data for this texture from the GPU.

        Parameters
        ----------
        shape : tuple[int, ...], optional
            The shape of the image to read. If not defined, then the shape will be the same as the original image.

        Returns
        -------
        numpy array of shape (height, width, channels)
            The texture data as a numpy array with the same width, height and channels as the original image.

        Notes
        -----
        The texture must be initialised before it can be read.

        Raises
        ------
        RuntimeError
            If the texture has not been initialised when the method is called.
        """
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(self.target, self.handle)

        # Get the colour data from the texture
        colour_string = glGetTexImage(
            self.target, 0, self.image_format, GL_UNSIGNED_BYTE
        )

        glBindTexture(self.target, 0)

        # Return the colour data as a numpy array
        self.__image = np.frombuffer(colour_string, dtype=np.uint8).reshape(
            self.__image.shape if shape is None else shape
        )
        return self.__image

    @require_initialised
    def update(
        self,
        image: Optional[
            Union[
                np.ndarray[(Any, Any, Any), np.dtype[np.uint8]],
                PILImage,
                MPImage,
            ]
        ] = None,
        size: Optional[tuple[int, int]] = None,
        alpha: bool = False,
    ):
        """Update the texture with new image data.

        Update the texture with new image data by providing either a new image or a new size. The new image must be
        either a numpy array, a PIL image or a MediaPipe image. If the image is not provided, then an empty texture
        will be used with the given size and alpha channel.

        Parameters
        ----------
        image : numpy array of shape (height, width, channels) or PIL image or MediaPipe image, optional
            The new image to update the texture with. If not defined, then the size parameter must be defined.
        size : tuple[int, int], optional
            The size of the new image. If not defined, then the image parameter must be defined.
        alpha : bool, optional
            Whether to include alpha in the new image. By default, this is False.

        Raises
        ------
        ValueError
            If both image and size are defined or if neither image nor size are defined or
            if the provided image is not a numpy array, PIL image or MediaPipe image.

        Notes
        -----
        If the image is not provided, then an empty texture will be stored with a given size.

        The given image is assumed to be in RGB or RGBA format. If the image is not in this format, then it must be
        converted before using this function.
        """
        # If the image and size are both None or both not None, then raise an error
        if (image is not None) == (size is not None):
            raise ValueError("Either img or size must be provided, but not both")

        # If the image is None, then create an empty image
        if image is None:
            assert size is not None
            image = np.zeros((size[1], size[0], 3 if not alpha else 4), dtype=np.uint8)

        if isinstance(image, np.ndarray):
            image = image
        elif isinstance(image, PILImage):
            image = np.array(image)
        elif isinstance(image, MPImage):
            image = image.numpy_view()
        else:
            raise ValueError("Invalid image type")

        # If the image has the same size and pixel format then we can use glTexSubImage2D
        # Otherwise, we need to use glTexImage2D
        same_shape_and_type = (
            image.shape == self.__image.shape and image.dtype == self.__image.dtype
        )

        self.__image = image.copy()
        image_pointer = self.__image.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))

        # If the image pointer is the same as the current image pointer, then return because we have not changed image
        if image_pointer == self.__image_pointer:
            return

        self.__image_pointer = image_pointer
        self.bind()

        if same_shape_and_type:
            glTexSubImage2D(
                self.target,
                0,
                0,
                0,
                self.width,
                self.height,
                self.image_format,
                GL_UNSIGNED_BYTE,
                self.__image_pointer,
            )
        else:
            glTexImage2D(
                self.target,
                0,
                self.image_format,
                self.width,
                self.height,
                0,
                self.image_format,
                GL_UNSIGNED_BYTE,
                self.__image_pointer,
            )

        self.unbind()

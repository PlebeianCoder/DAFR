from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field

from OpenGL.GL import *

from .texture import Texture
from .utils import Initialisable
from .utils import require_initialised


@dataclass
class RenderTarget(Initialisable):
    """
    A class to represent a render target in OpenGL.

    A render target is a framebuffer that we can draw to. It can be used to draw to a texture, which can then be used
    as a texture in a shader or can be read from.

    Note: to use the render target to draw to a texture, the texture must be set using the set_colour_buffer method.
    """

    # The following are private fields that store the framebuffer and renderbuffer handles and the viewport size
    __framebuffer_handle: GLuint = field(init=False, default=0, repr=False)
    __renderbuffer_handle: GLuint = field(init=False, default=0, repr=False)
    __viewport_width: int = field(init=False, default=-1, repr=False)
    __viewport_height: int = field(init=False, default=-1, repr=False)

    def __post_init__(self):
        # After initialisation, we must call the super class's __init__ method which setups the initialisation flag
        super().__init__()

    @property
    def fbo(self) -> GLuint:
        """
        :return: The framebuffer handle.
        """
        return self.__framebuffer_handle

    def init(self):
        """
        Initialise the render target, creating the FBO.
        """
        self.__framebuffer_handle = glGenFramebuffers(1)

        # Call the super class's init method to set the initialisation flag
        super().init()

    def exit(self):
        """
        Exit the render target, deleting the FBO.
        """
        super().exit()

        if self.__framebuffer_handle:
            glDeleteFramebuffers(1, [self.__framebuffer_handle])
            self.__framebuffer_handle = 0

    @require_initialised
    def set_colour_buffer(self, texture: Texture):
        """
        Set the colour buffer of the render target to the given texture.

        :param texture: The texture that the colour buffer will be written to.
        """

        # Bind the framebuffer and say that we will draw to the entire texture width and height
        glBindFramebuffer(GL_FRAMEBUFFER, self.__framebuffer_handle)
        glViewport(0, 0, texture.width, texture.height)

        # Bind the texture to the framebuffer
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(texture.target, texture.handle)
        glFramebufferTexture2D(
            GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, texture.target, texture.handle, 0
        )
        glBindTexture(texture.target, 0)

        # If the existing depth buffer has different dimensions, delete it
        if self.__renderbuffer_handle and (
            self.__viewport_width != texture.width
            or self.__viewport_height != texture.height
        ):
            glDeleteRenderbuffers(1, [self.__renderbuffer_handle])
            self.__renderbuffer_handle = 0

        # If there is no depth buffer, create one
        if not self.__renderbuffer_handle:
            # Create the render buffer
            self.__renderbuffer_handle = glGenRenderbuffers(1)
            # Bind the renderbuffer and allocate storage for it
            glBindRenderbuffer(GL_RENDERBUFFER, self.__renderbuffer_handle)
            glRenderbufferStorage(
                GL_RENDERBUFFER, GL_DEPTH_COMPONENT16, texture.width, texture.height
            )
            # Attach the renderbuffer to the framebuffer
            glFramebufferRenderbuffer(
                GL_FRAMEBUFFER,
                GL_DEPTH_ATTACHMENT,
                GL_RENDERBUFFER,
                self.__renderbuffer_handle,
            )
            # Unbind the renderbuffer
            glBindRenderbuffer(GL_RENDERBUFFER, 0)

        # Update the viewport width and height
        self.__viewport_width = texture.width
        self.__viewport_height = texture.height

        # Unbind the framebuffer
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        # Flush the OpenGL pipeline
        glFlush()

    @require_initialised
    def bind(self):
        """
        Bind the render target, binding the frame buffer and setting the viewport to the size of the render target.
        """
        glBindFramebuffer(GL_FRAMEBUFFER, self.__framebuffer_handle)
        glViewport(0, 0, self.__viewport_width, self.__viewport_height)

    @require_initialised
    def unbind(self):
        """
        Unbind the render target, unbinding the frame buffer.
        """
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

    @require_initialised
    def clear(self):
        """
        Clear the render target, clearing the colour and depth buffers
        """
        self.bind()

        glEnable(GL_DEPTH_TEST)
        glDepthMask(GL_TRUE)

        glClearColor(0, 0, 0, 0)
        glClearDepth(1)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glDepthMask(GL_FALSE)
        glDisable(GL_DEPTH_TEST)

        self.unbind()
        glFlush()

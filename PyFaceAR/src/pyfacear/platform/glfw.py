import glfw

from ..utils import require_initialised
from .base import Platform


class GLFWPlatform(Platform):
    """
    A class to manage an OpenGL context using GLFW.
    """

    def __init__(
        self, width: int, height: int, visible: bool = False, title: str = "PyFaceAR"
    ):
        """
        Initialise the OpenGL context.

        :param width: The width of the viewport
        :param height: The height of the viewport
        :param visible: Whether the viewport should be visible
        :param title: The title of the window
        """
        super().__init__(width, height)
        self.__visible = visible
        self.__title = title
        self.__window_handle = 0

    @property
    def visible(self):
        """
        :return: Whether the viewport is visible
        """
        return self.__visible

    @property
    def title(self):
        return self.__title

    @property
    def window_handle(self):
        """
        :return: The window handle
        """
        return self.__window_handle

    @require_initialised
    def resize(self, width: int, height: int):
        """
        Resize the window.

        :param width: The new width
        :param height: The new height
        """
        self._width = width
        self._height = height
        if self.__window_handle:
            glfw.set_window_size(self.__window_handle, width, height)

    def init(self):
        # Initialise GLFW
        if not glfw.init():
            raise RuntimeError("Failed to initialise GLFW")
        # Set the window to be invisible if necessary
        if not self.__visible:
            glfw.window_hint(glfw.VISIBLE, glfw.FALSE)

        # Create the window
        self.__window_handle = glfw.create_window(
            self._width, self._height, self.__title, None, None
        )

        # Terminate GLFW if the window could not be created
        if not self.__window_handle:
            glfw.terminate()
            raise RuntimeError("Failed to create window")

        # Make the window the current context
        glfw.make_context_current(self.__window_handle)
        # Call the super class's init method to set the initialisation flag
        super().init()

    def exit(self):
        # Call the super class's exit method to unset the initialisation flag
        super().exit()
        # Destroy the window if it exists
        if self.__window_handle:
            glfw.destroy_window(self.__window_handle)
            self.__window_handle = 0
        # Terminate GLFW
        glfw.terminate()

    @require_initialised
    def should_close(self) -> bool:
        """
        :return: Whether the window should close
        """
        return glfw.window_should_close(self.__window_handle)

    @require_initialised
    def swap_buffers(self):
        """
        Swap the front and back buffers.
        """
        glfw.swap_buffers(self.__window_handle)

    @require_initialised
    def poll_events(self):
        """
        Poll for events.
        """
        glfw.poll_events()

    @require_initialised
    def set_key_callback(self, callback):
        """
        Set the key callback.
        """
        glfw.set_key_callback(self.__window_handle, callback)

    @require_initialised
    def set_framebuffer_size_callback(self, callback):
        """
        Set the framebuffer size callback.
        """
        glfw.set_framebuffer_size_callback(self.__window_handle, callback)

    @require_initialised
    def set_drop_callback(self, callback):
        """
        Set the drop callback.
        """
        glfw.set_drop_callback(self.__window_handle, callback)

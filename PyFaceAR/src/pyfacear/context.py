from __future__ import annotations

from .utils import Initialisable
from .utils import require_initialised


def auto_detect_platform(
    width: int, height: int, visible: bool = False, title: str = "PyFaceAR"
):
    """
    Automatically detect the platform and create an OpenGL context.

    :param width: The width of the viewport
    :param height: The height of the viewport
    :param visible: Whether the viewport should be visible
    :param title: The title of the window
    :return: The platform
    """
    if not visible:
        # If we are not visible then first try to create an EGL platform
        try:
            from .platform.egl import EGLPlatform

            return EGLPlatform(width, height)
        except RuntimeError:
            pass
        except ImportError:
            pass
        # Otherwise, create a GLFW platform
        from .platform.glfw import GLFWPlatform

        return GLFWPlatform(width, height, visible, title)

    # If we are visible then create a GLFW platform
    from .platform.glfw import GLFWPlatform

    return GLFWPlatform(width, height, visible, title)


class OpenGLContext(Initialisable):
    """
    A class to represent the OpenGL context.
    """

    def __init__(
        self, width: int, height: int, visible: bool = False, title: str = "PyFaceAR"
    ):
        # After initialisation, we must call the super class's __init__ method which setups the initialisation flag
        super().__init__()
        self.__width = width
        self.__height = height
        self.__visible = visible
        self.__title = title
        self.__platform = auto_detect_platform(width, height, visible, title)

    @property
    def width(self):
        return self.__platform.width

    @property
    def height(self):
        return self.__platform.height

    @property
    def visible(self):
        return self.__visible

    @property
    def title(self):
        return self.__title

    def init(self):
        """
        Initialise the OpenGL context.
        """
        self.__platform.init()
        # Call the super class's init method to set the initialisation flag
        super().init()

    def exit(self):
        """
        Clean up the OpenGL context.
        """
        # Call the super class's exit method to reset the initialisation flag
        super().exit()
        self.__platform.exit()

    @require_initialised
    def resize(self, width: int, height: int):
        """
        Resize the context.

        :param width: The new width
        :param height: The new height
        """
        self.__platform.resize(width, height)

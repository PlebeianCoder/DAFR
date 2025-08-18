from abc import ABCMeta
from abc import abstractmethod

from ..utils import Initialisable
from ..utils import require_initialised


class Platform(Initialisable, metaclass=ABCMeta):
    """
    A class to manage a platform that can create an OpenGL context.
    """

    def __init__(self, width: int, height: int):
        """
        Initialise the platform.

        :param width: The width of the viewport
        :param height: The height of the viewport
        """
        super().__init__()
        self._width = width
        self._height = height

    @property
    def width(self) -> int:
        """
        :return: The width of the viewport
        """
        return self._width

    @property
    def height(self) -> int:
        """
        :return: The height of the viewport
        """
        return self._height

    @abstractmethod
    @require_initialised
    def resize(self, width: int, height: int):
        pass

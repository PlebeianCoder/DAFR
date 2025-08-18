"""Utility functions, decorators and classes used throughout PyFaceAR.

This module contains utility functions, decorators and classes that are used throughout PyFaceAR. This includes
initialisable objects, a decorator to require that a method is only called when an object is initialised.

Notes
-----
The `singleton` decorator can be used to ensure that a method is only called once. This is useful for initialising
objects that are expensive to create and are not expected to change.

The `Initialisable` class provides a standard interface for deferred initialisation of an object and specific
de-initialisation. This is useful for objects that require a specific initialisation sequence or need to be
de-initialised in a specific way and is commonly used for OpenGL objects.

The `require_initialised` decorator can then be used to ensure that a method is only called when the Initialisable
object has been initialised.
"""
from __future__ import annotations

from abc import ABCMeta
from abc import abstractmethod
from functools import lru_cache
from functools import wraps
from typing import Any

import numpy as np

singleton = lru_cache(maxsize=1)
"""
callable: A decorator that can be used to ensure that a method is only called once.

A decorator that can be used to ensure that a method is only called once. The result of the first call is cached and
returned for all subsequent calls.
"""


def require_initialised(func):
    """Require that a method is only called when the object has been initialised.

    A decorator that can be used to ensure that a method is only called when an initialisable object has been
    initialised using the `init()` method.

    Parameters
    ----------
    func : callable
        The method to ensure is only called when the object is initialised.

    Returns
    -------
    callable
        The decorated method that can only be called when the object is initialised.

    Raises
    ------
    RuntimeError
        If the object has not been initialised when the method is called.

    Notes
    -----
    The decorated function must be a non-static method of a class that inherits from Initialisable.

    Examples
    --------
    >>> class MyClass(Initialisable):
    ...     def __init__(self):
    ...         super().__init__()
    ...
    ...     def init(self):
    ...         super().init()
    ...
    ...     def exit(self):
    ...         super().exit()
    ...
    ...     @require_initialised
    ...     def my_method(self):
    ...         pass

    >>> obj = MyClass()
    >>> obj.my_method()
    Traceback (most recent call last):
    ...
    RuntimeError: MyClass is not initialised
    >>> obj.init()
    >>> obj.my_method()
    >>> # No error is raised
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        self = args[0]
        if not self.initialised:
            raise RuntimeError(f"{self.__class__.__name__} is not initialised")
        return func(*args, **kwargs)

    return wrapper


class Initialisable(metaclass=ABCMeta):
    """Object that has a deferred initialisation and specific de-initialisation.

    An abstract base class that provides a standard interface for deferred initialisation of an object and specific
    de-initialisation. This is useful for objects that require a specific initialisation sequence or need to be
    de-initialised in a specific way and is commonly used for OpenGL objects.

    Notes
    -----
    The `init` method should be called to initialise the object, and the `exit` method should be called to de-initialise
    the object. The `initialised` property can be used to check if the object is initialised.

    The `require_initialised` decorator can then be used to ensure that a method is only called when the object is
    initialised.

    To call the `init` method when entering a block and the `exit` method when leaving a block, the object can be used
    with the `with` statement.

    See Also
    --------
    require_initialised
    """

    def __init__(self):
        """Create an initialisable object, marking it as not initialised."""
        self.__initialised = False

    @abstractmethod
    def init(self):
        """Initialise the object, marking it as initialised."""
        self.__initialised = True

    @abstractmethod
    def exit(self):
        """De-initialise the object, marking it as not initialised."""
        self.__initialised = False

    @property
    def initialised(self) -> bool:
        """bool: True if the object is initialised, otherwise False."""
        return self.__initialised

    def __enter__(self):
        """Initialise the object when entering a `with` block.

        Returns
        -------
        The initialisable object that has been initialised.
        """
        self.init()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """De-initialise the object when leaving a `with` block."""
        self.exit()


def apply_mask(
    image: np.ndarray[(Any, Any, 3), Any], mask: [(Any, Any), np.dtype[np.bool_]]
) -> np.ndarray[(Any, Any, 4), Any]:
    """Apply a mask to an image, adding an alpha channel.

    Apply a mask to an image, where the mask is a boolean array with the same shape as the image.
    This will add an alpha channel to the image, where the alpha channel is 0 where the mask is False and 255 where the
    mask is True.

    Parameters
    ----------
    image : (H, W, 3) numpy.ndarray
        The image to apply the mask to.
    mask : (H, W) numpy.ndarray
        The mask to apply to the image.

    Returns
    -------
    (H, W, 4) numpy.ndarray
        The image with the mask applied.

    Raises
    ------
    ValueError
        If the image and mask have different shapes.
    """
    if image.shape[:2] != mask.shape:
        raise ValueError("The image and mask must have the same shape")
    return np.append(image, mask[:, :, np.newaxis].astype(np.uint8) * 255, axis=2)

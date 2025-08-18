"""
MIT License

Copyright (c) 2019 Matthew Matl

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
# This file is primarily based on the EGL platform from PyRenderer.
# The original file can be found at:
# https://github.com/mmatl/pyrender/blob/master/pyrender/platforms/egl.py
from __future__ import annotations

import ctypes
import os
from dataclasses import dataclass
from typing import Optional

import OpenGL.platform

from ..utils import require_initialised
from .base import Platform

EGL_PLATFORM_DEVICE_EXT = 0x313F
EGL_DRM_DEVICE_FILE_EXT = 0x3233


def _ensure_egl_loaded():
    """
    Ensure the EGL library is loaded.
    """
    plugin = OpenGL.platform.PlatformPlugin.by_name("egl")
    if plugin is None:
        raise RuntimeError("EGL plugin platform is not available")

    plugin_class = plugin.load()
    plugin.loaded = True
    # Create an instance of the plugin class
    plugin = plugin_class()
    # Install the plugin
    plugin.install(vars(OpenGL.platform))


_ensure_egl_loaded()
from OpenGL import EGL as egl


def _get_egl_func(
    func_name: str, res_type: type, *arg_types: type
) -> Optional[ctypes.CFUNCTYPE]:
    """
    Get an EGL function by name.

    :param func_name: The function name.
    :param res_type: The function return type.
    :param arg_types: The function argument types.
    :return: The EGL function.
    """
    address = egl.eglGetProcAddress(func_name)
    if address is None:
        return None

    proto = ctypes.CFUNCTYPE(res_type)
    proto.argtypes = arg_types
    return proto(address)


def _get_egl_struct(struct_name: str) -> type:
    """
    Get an EGL struct by name.

    :param struct_name: The name of the struct.
    :return: The EGL struct.
    """
    from OpenGL._opaque import opaque_pointer_cls

    return opaque_pointer_cls(struct_name)


# These are not defined in PyOpenGL by default.
_EGLDeviceEXT = _get_egl_struct("EGLDeviceEXT")
_eglGetPlatformDisplayEXT = _get_egl_func("eglGetPlatformDisplayEXT", egl.EGLDisplay)
_eglQueryDevicesEXT = _get_egl_func("eglQueryDevicesEXT", egl.EGLBoolean)
_eglQueryDeviceStringEXT = _get_egl_func("eglQueryDeviceStringEXT", ctypes.c_char_p)


def query_devices() -> list["EGLDevice"]:
    """
    Query the EGL devices.

    :return: The list of EGL devices that are available.
    """
    if _eglQueryDevicesEXT is None:
        raise RuntimeError("EGL query extension is not loaded or is not supported")

    # Get the number of devices
    device_count = egl.EGLint()
    success = _eglQueryDevicesEXT(0, None, ctypes.pointer(device_count))
    if not success or device_count.value < 1:
        return []

    # Create an array of size device_count to fetch all the devices
    devices = (_EGLDeviceEXT * device_count.value)()
    success = _eglQueryDevicesEXT(
        device_count.value, devices, ctypes.pointer(device_count)
    )
    if not success or device_count.value < 1:
        return []

    # Return the devices
    return [EGLDevice(dev) for dev in devices]


def get_default_device() -> "EGLDevice":
    """
    Get the default EGL device.

    :return: The default EGL device.
    """
    # Fall back to not using the extension
    if _eglQueryDevicesEXT is None:
        return EGLDevice(None)

    return query_devices()[0]


def get_device_by_index(device_id: int) -> "EGLDevice":
    """
    Get an EGL device by index.

    :param device_id: The device index.
    :return: The EGL device.
    """
    # Fall back to not using the extension
    if _eglQueryDevicesEXT is None and device_id == 0:
        return get_default_device()

    # Get the devices and check the index
    devices = query_devices()
    if device_id < 0 or device_id >= len(devices):
        raise ValueError(f"Device index {device_id} is out of range")
    return devices[device_id]


@dataclass
class EGLDevice:
    """
    A class to manage an EGL device.
    """

    device: _EGLDeviceEXT

    def get_display(self):
        """
        Get the EGL display for the device.
        """
        if self.device is None:
            return egl.eglGetDisplay(egl.EGL_DEFAULT_DISPLAY)
        return _eglGetPlatformDisplayEXT(EGL_PLATFORM_DEVICE_EXT, self.device, None)

    @property
    def name(self) -> str:
        """
        :return: The name of the device.
        """
        if self.device is None:
            return "default"

        name = _eglQueryDeviceStringEXT(self.device, EGL_DRM_DEVICE_FILE_EXT)
        if name is None:
            return "unknown"

        return name.decode("ascii")


class EGLPlatform(Platform):
    """
    A class to manage an OpenGL context using EGL.
    """

    def __init__(self, width: int, height: int, device: Optional[EGLDevice] = None):
        """
        Initialise the EGL platform.

        :param width: The width of the viewport.
        :param height: The height of the viewport.
        :param device: The EGL device to use. Using None will use the default device.
        """
        super().__init__(width, height)
        if device is None:
            device = get_default_device()

        self._egl_device = device
        self._egl_display = None
        self._egl_context = None

    def init(self):
        """
        Initialise the EGL platform.
        """
        from OpenGL.EGL import (
            EGL_SURFACE_TYPE,
            EGL_PBUFFER_BIT,
            EGL_BLUE_SIZE,
            EGL_RED_SIZE,
            EGL_GREEN_SIZE,
            EGL_DEPTH_SIZE,
            EGL_COLOR_BUFFER_TYPE,
            EGL_RGB_BUFFER,
            EGL_RENDERABLE_TYPE,
            EGL_OPENGL_BIT,
            EGL_CONFORMANT,
            EGL_NONE,
            EGL_NO_CONTEXT,
            EGL_OPENGL_API,
            EGL_CONTEXT_MAJOR_VERSION,
            EGL_CONTEXT_MINOR_VERSION,
            EGL_CONTEXT_OPENGL_PROFILE_MASK,
            EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT,
            eglInitialize,
            eglChooseConfig,
            eglBindAPI,
            eglCreateContext,
            EGLConfig,
            eglMakeCurrent,
            EGL_NO_TEXTURE,
            EGL_NO_SURFACE,
        )
        from OpenGL import arrays

        config_attributes = arrays.GLintArray.asArray(
            [
                EGL_SURFACE_TYPE,
                EGL_PBUFFER_BIT,
                EGL_BLUE_SIZE,
                8,
                EGL_RED_SIZE,
                8,
                EGL_GREEN_SIZE,
                8,
                EGL_DEPTH_SIZE,
                24,
                EGL_COLOR_BUFFER_TYPE,
                EGL_RGB_BUFFER,
                EGL_RENDERABLE_TYPE,
                EGL_OPENGL_BIT,
                EGL_CONFORMANT,
                EGL_OPENGL_BIT,
                EGL_NONE,
            ]
        )
        context_attributes = arrays.GLintArray.asArray(
            [
                EGL_CONTEXT_MAJOR_VERSION,
                4,
                EGL_CONTEXT_MINOR_VERSION,
                1,
                EGL_CONTEXT_OPENGL_PROFILE_MASK,
                EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT,
                EGL_NONE,
            ]
        )
        major, minor = ctypes.c_long(), ctypes.c_long()
        num_configs = ctypes.c_long()
        configs = (EGLConfig * 1)()

        # Cache DISPLAY if necessary and get an off-screen EGL display
        orig_dpy = None
        if "DISPLAY" in os.environ:
            orig_dpy = os.environ["DISPLAY"]
            del os.environ["DISPLAY"]

        self._egl_display = self._egl_device.get_display()
        if orig_dpy is not None:
            os.environ["DISPLAY"] = orig_dpy

        # Initialize EGL
        assert eglInitialize(self._egl_display, major, minor)
        assert eglChooseConfig(
            self._egl_display, config_attributes, configs, 1, num_configs
        )

        # Bind EGL to the OpenGL API
        assert eglBindAPI(EGL_OPENGL_API)

        # Create an EGL context
        self._egl_context = eglCreateContext(
            self._egl_display, configs[0], EGL_NO_CONTEXT, context_attributes
        )

        # Make EGL the current context
        assert eglMakeCurrent(
            self._egl_display, EGL_NO_SURFACE, EGL_NO_SURFACE, self._egl_context
        )

        super().init()

    def exit(self):
        """
        Clean up the EGL platform.
        """
        super().exit()

        from OpenGL.EGL import eglDestroyContext, eglTerminate

        if self._egl_display is not None:
            if self._egl_context is not None:
                eglDestroyContext(self._egl_display, self._egl_context)
                self._egl_context = None
            eglTerminate(self._egl_display)
            self._egl_display = None

    @require_initialised
    def resize(self, width: int, height: int):
        self._width = width
        self._height = height

        # For EGL we don't need to do anything here as EGL does not have a window

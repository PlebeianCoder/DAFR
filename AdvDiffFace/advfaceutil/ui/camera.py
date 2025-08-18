from logging import getLogger
from pathlib import Path
from threading import Event as ThreadEvent
from threading import Thread
from time import strftime
from tkinter import *
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional

import cv2
import numpy as np
from PIL import Image as PImage
from PIL import ImageTk
from pygrabber.dshow_graph import FilterGraph
from pygrabber.dshow_graph import VideoInput

from advfaceutil.ui.trace import trace


LOGGER = getLogger("camera")


class Camera(Canvas):
    @staticmethod
    def _try_get_formats(video_input: VideoInput) -> List[Dict[str, Any]]:
        try:
            return video_input.get_formats()
        except KeyError:
            return [{"width": 1280, "height": 720, "max_framerate": 30}]

    @staticmethod
    def _get_available_cameras() -> Dict[int, VideoInput]:
        graph = FilterGraph()
        devices = graph.get_input_devices()
        available_cameras = {}
        for device_index, _ in enumerate(devices):
            graph.add_video_input_device(device_index)
            video_input = graph.get_input_device()
            available_cameras[device_index] = video_input
            graph.remove_filters()
        return available_cameras

    def __init__(
        self,
        master: Optional[Misc],
        camera_processing: Callable[[cv2.typing.MatLike], cv2.typing.MatLike],
        *args,
        **kwargs,
    ):
        super().__init__(master, *args, **kwargs, background="black")
        self._camera_processing = camera_processing

        self._image = None
        self.__video: Optional[cv2.VideoCapture] = None
        self.__camera_image = None
        self.__image_output = None
        self.__available_cameras: Dict[int, VideoInput] = self._get_available_cameras()
        if len(self.__available_cameras) > 0:
            # If we have available cameras then load the format for the first one
            self.__available_formats = self.__available_formats = self._try_get_formats(
                self.__available_cameras[0]
            )
        else:
            self.__available_formats = []
        self.__camera_aspect = 1
        self.__resized_camera_size = None
        self.__camera_streaming_thread = None
        self.__stop_camera = ThreadEvent()
        self.__camera_stopped = ThreadEvent()

        self.selected_camera = IntVar(self, 0, name="selected_camera")
        self.selected_resolution = StringVar(self, name="selected_resolution")
        self.selected_framerate = IntVar(self, 30, name="selected_framerate")
        self.is_active = BooleanVar(self, False, name="is_active")

        self.bind("<Configure>", self._resize)

    @trace("selected_framerate", "write")
    def _update_framerate(self):
        if not self._is_active:
            return
        self.__video.set(cv2.CAP_PROP_FPS, self.selected_framerate.get())

    def _read_from_camera(
        self, stop_camera_event: ThreadEvent, camera_stopped_event: ThreadEvent
    ):
        camera_name = self.__available_cameras[self.selected_camera.get()].Name
        LOGGER.info("Starting to read from camera '%s'", camera_name)
        camera_stopped_event.clear()

        while not stop_camera_event.is_set():
            if not self._is_active:
                return

            result, frame = self.__video.read()
            if not result:
                if stop_camera_event.is_set():
                    break
                # sleep(0.1)
                continue

            camera_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.__camera_image = camera_image = self._camera_processing(camera_image)

            if self.__resized_camera_size:
                camera_image = cv2.resize(
                    camera_image,
                    self.__resized_camera_size,
                    interpolation=cv2.INTER_NEAREST,
                )

            captured_image = PImage.fromarray(camera_image)
            photo_image = ImageTk.PhotoImage(image=captured_image)

            if self.__image_output is not None:
                self.itemconfigure(self.__image_output, image=photo_image)
                self.moveto(
                    self.__image_output,
                    (self.winfo_width() - photo_image.width()) // 2,
                    (self.winfo_height() - photo_image.height()) // 2,
                )
            else:
                self.__image_output = self.create_image(
                    0, 0, image=photo_image, anchor=NW
                )

            self._image = photo_image
        camera_stopped_event.set()
        stop_camera_event.clear()
        LOGGER.info("Finished reading from camera '%s'", camera_name)

    def _resize(self, _):
        self._calculate_resized_camera_size()

    @trace("selected_camera", "write")
    def _selected_camera_updated(self):
        # Find the new camera and ensure it is valid
        selected_camera = self.selected_camera.get()
        if selected_camera not in self.__available_cameras.keys():
            self.__available_formats.clear()
            return

        # Load the available camera formats
        self.__available_formats = self._try_get_formats(
            self.__available_cameras[selected_camera]
        )
        LOGGER.info(
            "Switching to device %s", self.__available_cameras[selected_camera].Name
        )

        # Restart the camera
        self.stop_camera()
        self._try_restart_camera()

    def _try_restart_camera(self):
        if not self.__camera_stopped.is_set():
            self.after_idle(self._try_restart_camera)
        else:
            self.start_camera()

    @trace("selected_resolution", "write")
    def _selected_resolution_updated(self):
        # We can only update the resolution if we are active
        if not self._is_active:
            return
        assert self.selected_resolution.get() in self.available_resolutions
        width, height = self.selected_resolution.get().split("x")
        self.__video.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
        self.__video.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
        self.__camera_aspect = int(width) / int(height)
        self._calculate_resized_camera_size()

    def _calculate_resized_camera_size(self):
        # We can only update the resolution if we are active
        if not self._is_active:
            return
        width, height = self.winfo_width(), self.winfo_height()

        new_width = min(np.round(height * self.__camera_aspect).astype(int), width)
        new_height = min(np.round(new_width / self.__camera_aspect).astype(int), height)

        self.__resized_camera_size = (new_width, new_height)

    @property
    def available_cameras(self) -> Dict[int, str]:
        return {
            index: video_input.Name
            for index, video_input in self.__available_cameras.items()
        }

    @property
    def available_resolutions(self) -> List[str]:
        sorted_resolutions = sorted(
            set((f.get("width"), f.get("height")) for f in self.__available_formats),
            key=lambda r: r[0] * r[1],
            reverse=True,
        )
        return ["x".join(map(str, resolution)) for resolution in sorted_resolutions]

    @property
    def available_framerates(self) -> List[int]:
        return sorted(
            set(int(f.get("max_framerate")) for f in self.__available_formats),
            reverse=True,
        )

    @property
    def available_codecs(self) -> List[str]:
        return list(set(f.get("media_type_str") for f in self.__available_formats))

    @property
    def _is_active(self):
        return self.__video is not None

    def refresh_available_cameras(self):
        self.__available_cameras = self._get_available_cameras()

        # If the selected camera is no longer in the available cameras then reset the format
        selected_camera = self.selected_camera.get()
        if selected_camera not in self.__available_cameras.keys():
            self.__available_formats.clear()
            return
        # Otherwise get the available formats for the selected camera
        self.__available_formats = self._try_get_formats(
            self.__available_cameras[selected_camera]
        )

    def switch_camera(self, index: int):
        if self.selected_camera.get() == index:
            return
        if self.__stop_camera.is_set() and not self.__camera_stopped.is_set():
            LOGGER.error(
                "Could not switch camera to '%s' since the camera is trying to stop",
                self.__available_cameras[index].Name,
            )
            return
        self.selected_camera.set(index)

    def start_camera(self) -> bool:
        # If there is a camera already active then don't do anything
        if self.__video is not None:
            return False
        # Find the camera to start
        selected_camera = self.selected_camera.get()
        if selected_camera not in self.__available_cameras.keys():
            return False
        # Try to start the camera
        self.__video = cv2.VideoCapture(selected_camera, cv2.CAP_DSHOW)
        # Handle when we cannot return
        if not self.__video.isOpened():
            self.__video = None
            return False
        self.__video.set(
            cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.available_codecs[0])
        )
        self.is_active.set(True)
        # Set the resolution
        assert len(self.available_resolutions) > 0
        self.selected_resolution.set(self.available_resolutions[0])
        # Set the framerate
        assert len(self.available_framerates) > 0
        self.selected_framerate.set(self.available_framerates[0])
        # Start reading from the camera
        self.__camera_streaming_thread = Thread(
            target=self._read_from_camera,
            name="Camera Streaming",
            args=(
                self.__stop_camera,
                self.__camera_stopped,
            ),
            daemon=True,
        )
        self.__camera_streaming_thread.start()
        return True

    def stop_camera(self):
        # If there is no camera active then don't do anything
        if self.__video is None:
            return
        self.__stop_camera.set()
        self.is_active.set(False)
        self.__video.release()
        self.__video = None

    def capture_photo(self) -> Optional[Path]:
        if not self._is_active or self.__camera_image is None:
            return
        screenshot_path = Path("screenshot_" + strftime("%Y.%m.%d-%H.%M.%S") + ".png")
        cv2.imwrite(
            screenshot_path.as_posix(),
            cv2.cvtColor(self.__camera_image, cv2.COLOR_RGB2BGR),
        )
        return screenshot_path

__all__ = ["VaryLighting", "VaryLightingProperties"]

from typing import Generator, Tuple

import cv2
import numpy as np

from advfaceutil.benchmark.args import BenchmarkArguments
from advfaceutil.benchmark.augmentation.base import Augmentation, AugmentationFactory
from advfaceutil.benchmark.data import BenchmarkData, DataPropertyEnum, Accessory


class VaryLightingProperties(DataPropertyEnum):
    BRIGHTNESS_CHANGE = "brightness_change"
    CONTRAST_CHANGE = "contrast_change"


class VaryLighting(Augmentation):
    class Factory(AugmentationFactory):
        def __init__(
            self,
            brightness_range: Tuple[float, float] = (0.5, 1.5),
            contrast_range: Tuple[float, float] = (0.5, 1.5),
            repeat: int = 1,
        ):
            self.brightness_range = brightness_range
            self.contrast_range = contrast_range
            self.repeat = repeat

        @staticmethod
        def name() -> str:
            return "VaryLighting"

        def construct(
            self,
            benchmark_arguments: BenchmarkArguments,
            accessory: Accessory,
        ) -> "VaryLighting":
            return VaryLighting(
                self.name(),
                benchmark_arguments,
                accessory,
                self.brightness_range,
                self.contrast_range,
                self.repeat,
            )

    def __init__(
        self,
        name: str,
        benchmark_arguments: BenchmarkArguments,
        accessory: Accessory,
        brightness_range: Tuple[float, float],
        contrast_range: Tuple[float, float],
        repeat: int,
    ):
        super().__init__(name, benchmark_arguments, accessory)
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.repeat = repeat

    def __str__(self):
        return f"{self.name}(brightness_range={self.brightness_range},contrast_range={self.contrast_range})"

    def load_extra_data_from_data(
        self, data: BenchmarkData
    ) -> Generator[BenchmarkData, None, None]:
        if self.repeat <= 1:
            return

        for _ in range(self.repeat - 1):
            yield self.pre_augmentation_processing(data.copy())

    def extra_data_count_from_loaded_data(self, loaded_data: int) -> int:
        return loaded_data * (self.repeat - 1)

    def pre_augmentation_processing(self, data: BenchmarkData) -> BenchmarkData:
        # Vary the brightness of the image
        if (
            not data.has_property(VaryLightingProperties.BRIGHTNESS_CHANGE)
            and self.brightness_range[0] != self.brightness_range[1]
        ):
            hsv_image = cv2.cvtColor(data.image, cv2.COLOR_BGR2HSV)
            brightness = np.random.uniform(*self.brightness_range)
            hsv_image[:, :, 2] = np.clip(
                hsv_image[:, :, 2] * brightness, 0, 255
            ).astype(np.uint8)
            data.image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

            data.add_property(VaryLightingProperties.BRIGHTNESS_CHANGE, brightness)

        # Vary the contrast of the image
        if (
            not data.has_property(VaryLightingProperties.CONTRAST_CHANGE)
            and self.contrast_range[0] != self.contrast_range[1]
        ):
            contrast = np.random.uniform(*self.contrast_range)
            data.image = np.clip(
                contrast * (data.image - 127.5) + 127.5, 0, 255
            ).astype(np.uint8)

            data.add_property(VaryLightingProperties.CONTRAST_CHANGE, contrast)

        return data

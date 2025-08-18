__all__ = ["MatchLighting"]

import cv2
import numpy as np

from advfaceutil.benchmark.augmentation.base import Augmentation
from advfaceutil.benchmark.factory import default_factory
from advfaceutil.benchmark.data import BenchmarkData


# Be warned that the code does not work for SAS and SAS_DIFF processors.
@default_factory("MatchLighting")
class MatchLighting(Augmentation):
    def post_augmentation_processing(self, data: BenchmarkData) -> BenchmarkData:
        # Match the brightness of the mask to the image
        image_mask = data.augmented_image - data.image

        # Create a binary mask of where the accessory
        binary_mask = np.zeros(
            (image_mask.shape[0], image_mask.shape[1]), dtype=np.float64
        )
        binary_mask[np.any(image_mask != 0, axis=2)] = 1
        # Reshape the mask to be three channels
        binary_mask = binary_mask[:, :, np.newaxis]

        # Mask the image using the binary mask
        mask = binary_mask * data.augmented_image
        mask = mask.astype(np.uint8)

        hsv_image = cv2.cvtColor(data.image, cv2.COLOR_RGB2HSV).astype(np.float64)
        hsv_mask = cv2.cvtColor(mask, cv2.COLOR_RGB2HSV).astype(np.float64)

        brightness_difference = (hsv_image[:, :, 2] - hsv_image[:, :, 2].min()) / (
            hsv_image[:, :, 2].max() - hsv_image[:, :, 2].min()
        )
        hsv_mask[:, :, 2] *= np.mean(brightness_difference)

        hsv_mask[:, :, 2] = np.clip(hsv_mask[:, :, 2], 0, 255)
        hsv_mask = hsv_mask.astype(np.uint8)

        converted_mask = cv2.cvtColor(hsv_mask, cv2.COLOR_HSV2RGB)
        data.augmented_image = data.image * (1 - binary_mask) + converted_mask
        data.augmented_image = np.clip(data.augmented_image, 0, 255).astype(np.uint8)

        return data

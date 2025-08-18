__all__ = ["OcclusionOptions", "occlude", "Occlusion"]

from enum import Enum
from typing import Tuple
from random import choices, random

import torch
import torch.nn as nn


class OcclusionOptions(Enum):
    NONE = (60, (0, 0), (0, 0))
    LOWER = (30, (0, 0.5), (1, 1))
    UPPER = (4, (0, 0), (1, 0.5))
    LEFT = (3, (0, 0), (0.5, 1))
    RIGHT = (3, (0.5, 0), (1, 1))

    def __init__(
        self, weight: float, left: Tuple[float, float], right: Tuple[float, float]
    ) -> None:
        self.weight = weight
        self.left = left
        self.right = right

    @classmethod
    def random(cls) -> "OcclusionOptions":
        return choices(
            list(cls), weights=list(map(lambda option: option.weight, cls)), k=1
        )[0]

    def occlude(self, image: torch.Tensor) -> torch.Tensor:
        x1, y1 = self.left
        x2, y2 = self.right

        width = image.shape[len(image.shape) - 1]
        height = image.shape[len(image.shape) - 2]

        x1 *= width
        x2 *= width
        y1 *= height
        y2 *= height

        random_colour = random()

        if image.ndim == 2:
            image[int(y1) : int(y2), int(x1) : int(x2)] = random_colour
        elif image.ndim == 3:
            image[:, int(y1) : int(y2), int(x1) : int(x2)] = random_colour
        elif image.ndim == 4:
            image[:, :, int(y1) : int(y2), int(x1) : int(x2)] = random_colour

        return image


def occlude(images: torch.Tensor) -> torch.Tensor:
    if images.ndim < 3:
        return OcclusionOptions.random().occlude(images)

    occluded_images = torch.zeros_like(images)
    for i in range(images.shape[0]):
        occluded_images[i] = OcclusionOptions.random().occlude(images[i])

    return occluded_images


class Occlusion(nn.Module):
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image (Tensor): Image to occlude.

        Returns:
            Tensor: Randomly occluded image.
        """
        return occlude(image)

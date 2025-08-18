__all__ = [
    "load_image",
    "save_image",
    "log_image",
    "has_alpha",
    "normalise_image",
    "unnormalise_image",
    "load_overlay",
]

from pathlib import Path
from typing import Optional
from typing import Union

import cv2
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import wandb
from PIL import Image


def load_image(path: Union[str, Path], size: Optional[int]) -> torch.Tensor:
    image = Image.open(path).convert("RGB")

    # Define a transformation to resize the image and convert it to a tensor
    if size is not None:
        transform = transforms.Compose(
            [
                transforms.Resize((size, size)),
                transforms.ToTensor(),
            ]
        )
    else:
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

    return transform(image)


def save_image(
    image: Union[torch.Tensor, np.ndarray],
    path: Union[str, Path],
    to_01: bool = False,
    normalise: bool = False,
) -> None:
    if isinstance(image, np.ndarray):
        image = torch.from_numpy(image)
    if to_01:
        torchvision.utils.save_image((image + 1) / 2, path, normalize=normalise)
    else:
        torchvision.utils.save_image(image, path, normalize=normalise)


def log_image(
    image: Union[torch.Tensor, np.ndarray], caption: str, bgr: bool = True
) -> None:
    if isinstance(image, torch.Tensor):
        grid = torchvision.utils.make_grid(image)
        # Add 0.5 after unnormalising to [0, 255] to round to the nearest integer
        image = (
            grid.mul(255)
            .add_(0.5)
            .clamp_(0, 255)
            .permute(1, 2, 0)
            .to("cpu", torch.uint8)
            .numpy()
        )
    if image.dtype == np.float32:
        image = unnormalise_image(normalise_image(image))
    if bgr:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    im = Image.fromarray(image)
    wandb.log({caption: wandb.Image(im)})


def has_alpha(image: np.ndarray) -> bool:
    return image.shape[2] == 4


def normalise_image(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32)
    for i in range(image.shape[2]):
        if np.max(image[:, :, i]) > 1:
            image[:, :, i] = image[:, :, i] / 255.0
    return image


def unnormalise_image(image: np.ndarray) -> np.ndarray:
    for i in range(image.shape[2]):
        image[:, :, i] = image[:, :, i] * 255.0
    return image.clip(0, 255).astype(np.uint8)


def load_overlay(path: Union[str, Path]) -> np.ndarray:
    """
    Load an overlay image from the given path.
    The image can be either a numpy file saved using an attack or an image file.

    :param path: The path to the overlay image.
    :return: The loaded image in RGB(A) with the channel last.
    """
    path = Path(path)
    if path.name.endswith(".npy"):
        # If we normalise and then unnormalise the image, we should get the image to be between 0 and 255
        return unnormalise_image(
            normalise_image(np.transpose(np.load(path), (1, 2, 0)))
        )
    return np.array(Image.open(path).convert("RGBA"))

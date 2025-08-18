__all__ = ["RecognitionArchitecture"]

from abc import ABCMeta
from abc import abstractmethod
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn

from advfaceutil.datasets import FaceDatasets
from advfaceutil.datasets import FaceDatasetSize


class RecognitionArchitecture(nn.Module, metaclass=ABCMeta):
    @staticmethod
    @abstractmethod
    def construct(
        dataset: FaceDatasets,
        size: FaceDatasetSize,
        weights_directory: Optional[Path] = None,
        training: bool = False,
        device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
    ) -> "RecognitionArchitecture":
        pass

    @staticmethod
    def preprocess(image: np.ndarray, toBGR=False, batched=False) -> torch.Tensor:
        if isinstance(image, list):
            image = image[0]
        # print("pre")
        # print(image)
        # print(image.shape)
        if toBGR:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # print(np.max(image))
        image = np.transpose(image.astype(np.float32), (2, 0, 1))
        image = image / 255.0
        if batched:
            return torch.unsqueeze(torch.from_numpy(image), 0)
        return torch.from_numpy(image)

    def logits(
        self,
        image: np.ndarray,
        device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
    ) -> torch.Tensor:
        preprocessed_image = self.preprocess(
            image, not hasattr(self, "noBGR"), hasattr(self, "batched")
        )
        preprocessed_image = preprocessed_image.to(device)
        outputs = self(preprocessed_image)
        return outputs

    def classify(
        self,
        image: np.ndarray,
        device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
    ) -> int:
        outputs = self.logits(image, device)
        return outputs.argmax().item()

    @abstractmethod
    def save_transfer_data(
        self, save_directory: Path, dataset: FaceDatasets, size: FaceDatasetSize
    ) -> None:
        pass

    @abstractmethod
    def load_transfer_data(
        self,
        weights_directory: Path,
        dataset: FaceDatasets,
        size: FaceDatasetSize,
        device: torch.device,
    ) -> None:
        pass

    crop_size = 112

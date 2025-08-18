__all__ = ["RecognitionArchitectures"]

from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Type, Optional

import torch

from advfaceutil.datasets import FaceDatasets
from advfaceutil.datasets import FaceDatasetSize
from advfaceutil.recognition.base import RecognitionArchitecture
from advfaceutil.recognition.clip import FaRL
from advfaceutil.recognition.insightface import IResNet
from advfaceutil.recognition.iresnethead import IResNetHead
from advfaceutil.recognition.mobilefacenet import MobileFaceNet


class RecognitionArchitectures(Enum):
    IRESNET = (IResNet,)
    IRESNETHEAD = (IResNetHead,)
    FARL = (FaRL,)
    MFN = (MobileFaceNet,)

    def __init__(self, architecture: Type[RecognitionArchitecture]):
        self.__architecture = architecture

    @lru_cache
    def construct(
        self,
        dataset: FaceDatasets,
        size: FaceDatasetSize,
        weights_directory: Optional[Path] = None,
        training: bool = False,
        device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
    ) -> RecognitionArchitecture:
        return self.__architecture.construct(
            dataset, size, weights_directory, training, device
        )

    @property
    def crop_size(self) -> int:
        return self.__architecture.crop_size

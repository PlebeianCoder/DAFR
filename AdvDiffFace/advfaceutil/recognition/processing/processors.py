__all__ = ["FaceProcessors"]

from enum import Enum
from functools import lru_cache

from typing import Type

from advfaceutil.recognition.processing.base import FaceProcessor
from advfaceutil.recognition.processing.dlib import DlibFaceProcessor
from advfaceutil.recognition.processing.mediapipe import MediaPipeFaceProcessor
from advfaceutil.recognition.processing.advmask import AdvMaskFaceProcessor
from advfaceutil.recognition.processing.advmask_diff import AdvDiffMaskFaceProcessor


class FaceProcessors(Enum):
    """
    Enumerate the available face processors.
    """

    DLIB = (DlibFaceProcessor,)
    MEDIAPIPE = (MediaPipeFaceProcessor,)
    ADV = (AdvMaskFaceProcessor,)
    ADV_DIFF = (AdvDiffMaskFaceProcessor,)

    def __init__(self, processor: Type[FaceProcessor]):
        self.__processor = processor

    @lru_cache
    def construct(self) -> FaceProcessor:
        """
        Construct a face processor of this type, caching the result.

        :return: The constructed face processor.
        """
        return self.__processor()

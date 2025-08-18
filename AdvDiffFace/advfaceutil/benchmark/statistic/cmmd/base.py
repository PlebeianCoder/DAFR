__all__ = ["CMMD"]

from functools import lru_cache, cached_property
from typing import Any
from importlib.resources import path

import numpy as np

from advfaceutil.benchmark.data import Accessory, DataPropertyEnum, DataBin
from advfaceutil.benchmark.statistic.base import AccessoryStatistic
from advfaceutil.benchmark.statistic.cmmd.distance import mmd
from advfaceutil.benchmark.statistic.cmmd.embedding import ClipEmbeddingModel
from advfaceutil.benchmark.statistic.cmmd.io import (
    compute_embeddings_for_image,
)

_BATCH_SIZE = 32


class CMMDProperties(DataPropertyEnum):
    CMMD_ACCESSORY = "cmmd_accessory"
    CMMD_FACENESS = "cmmd_faceness"


@lru_cache(maxsize=1)
def get_embedding_model() -> ClipEmbeddingModel:
    return ClipEmbeddingModel()


class CMMD(AccessoryStatistic):
    @staticmethod
    def name() -> str:
        return "CMMD"

    @cached_property
    def _face_dataset_embeddings(self) -> np.ndarray:
        with path(
            "advfaceutil.benchmark.statistic.cmmd", "face_dataset_embeddings.npy"
        ) as embedding_path:
            return np.load(embedding_path)

    def record_statistic(
        self,
        accessory: Accessory,
    ) -> None:
        # Compute the embeddings for the non-adversarial accessory
        non_adversarial_accessory_embedding = compute_embeddings_for_image(
            accessory.non_adversarial_accessory_path, get_embedding_model()
        )

        # Compute the embeddings for the adversarial accessory
        adversarial_accessory_embedding = compute_embeddings_for_image(
            accessory.adversarial_accessory_path, get_embedding_model()
        )

        accessory.add_property(
            CMMDProperties.CMMD_ACCESSORY,
            mmd(non_adversarial_accessory_embedding, adversarial_accessory_embedding),
        )
        accessory.add_property(
            CMMDProperties.CMMD_FACENESS,
            mmd(self._face_dataset_embeddings, adversarial_accessory_embedding),
        )

    def collate_statistics(self, data_bin: DataBin[Accessory]) -> Any:
        return {
            "accessory": self._collate_list_statistics(
                data_bin, CMMDProperties.CMMD_ACCESSORY
            ),
            "faceness": self._collate_list_statistics(
                data_bin, CMMDProperties.CMMD_FACENESS
            ),
        }

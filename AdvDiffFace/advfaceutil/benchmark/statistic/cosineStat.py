__all__ = ["CosineStat", "CosineStatProperties"]

from typing import Any

import numpy as np
import torch
from PIL import Image
import glob
import os

from advfaceutil.benchmark.factory import default_factory
from advfaceutil.benchmark.data import (
    BenchmarkData,
    BenchmarkProperties,
    DataPropertyEnum,
    DataBin,
    construct_bins_from_properties,
)
from advfaceutil.benchmark.statistic.base import Statistic, StatisticFactory
from advfaceutil.datasets import FaceDatasets
from advfaceutil.recognition.insightface import IResNet
from torch.nn import CosineSimilarity

device = torch.device("cuda")


class CosineStatProperties(DataPropertyEnum):
    COSINE_SIM = "cosine_sim"


class CosineStat(Statistic):
    class Factory(StatisticFactory):
        def __init__(
            self,
        ):
            pass

        @staticmethod
        def name() -> str:
            return "CosineStat"

        def construct(self, benchmark_arguments, accessory) -> "CosineStat":
            return CosineStat(self.name(), benchmark_arguments, accessory)

    # def __init____init__(
    #     self,
    #     name: str,
    #     benchmark_arguments,
    #     accessory,
    # ):
    #     super().__init__(name, benchmark_arguments, accessory)

    def record_statistic(self, data: BenchmarkData) -> None:
        with torch.no_grad():
            # Need to set threshold then load embedder
            torchLogit = torch.from_numpy(data.augmented_logits).to(device)
            faceClasses = self._benchmark_arguments.size.class_names
            if not hasattr(self, "anchor"):
                wd = str(self._benchmark_arguments.weights_directory)
                rNetstr = "r18"

                if "r34" in wd:
                    rNetstr = "r34"

                elif "r50" in wd:
                    rNetstr = "r50"

                elif "r100" in wd:
                    rNetstr = "r100"

                elif "fted100" in wd:
                    rNetstr = "fted100"

                elif "clip" in wd:
                    rNetstr = "farl"

                elif "mobilefacenet" in wd:
                    rNetstr = "mfn"

                datasetStr = "PUBFIG"
                if self._benchmark_arguments.dataset == FaceDatasets.VGGFACE2:
                    datasetStr = "VGGFACE2"

                load_name = self._accessory.target_class
                if load_name is None:
                    load_name = self._accessory.base_class
                self.anchor = torch.load(
                    f"../../../../anchors/masked_{rNetstr}_{datasetStr}_{load_name}.pth",
                    map_location=device,
                )
                self.cos_sim = CosineSimilarity()

            sim_measure = self.cos_sim(torchLogit, self.anchor).item()
            print(sim_measure)
            data.add_property(CosineStatProperties.COSINE_SIM, sim_measure)

    def collate_statistics(self, data_bin) -> Any:
        return self._collate_list_statistics(data_bin, CosineStatProperties.COSINE_SIM)

__all__ = ["CosineSuccessRate", "CosineSuccessRateProperties"]

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


class CosineSuccessRateProperties(DataPropertyEnum):
    # Both FAR of 0.01
    COSINE_100_SUCC = "cosine_100_succ"
    COSINE_1000_SUCC = "cosine_1000_succ"


class CosineSuccessRate(Statistic):
    class Factory(StatisticFactory):
        def __init__(
            self,
        ):
            pass

        @staticmethod
        def name() -> str:
            return "CosineSuccessRate"

        def construct(self, benchmark_arguments, accessory) -> "CosineSuccessRate":
            return CosineSuccessRate(self.name(), benchmark_arguments, accessory)

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
                self.threshold_100 = 0.2670
                self.threshold_1000 = 0.3027
                if "r34" in wd:
                    rNetstr = "r34"
                    self.threshold_100 = 0.2292
                    self.threshold_1000 = 0.2712

                elif "r50" in wd:
                    rNetstr = "r50"
                    self.threshold_100 = 0.2394
                    self.threshold_1000 = 0.2519

                elif "r100" in wd:
                    rNetstr = "r100"
                    self.threshold_100 = 0.2687
                    self.threshold_1000 = 0.2370

                elif "fted100" in wd:
                    rNetstr = "fted100"
                    self.threshold_100 = 0.5300
                    self.threshold_1000 = 0.8355

                elif "clip" in wd:
                    rNetstr = "farl"
                    self.threshold_100 = 0.7684
                    self.threshold_1000 = 0.7657
                elif "mobilefacenet" in wd:
                    rNetstr = "mfn"
                    self.threshold_100 = 0.6156
                    self.threshold_1000 = 0.6622

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
            if self._accessory.target_class is None:
                # Dodging
                data.add_property(
                    CosineSuccessRateProperties.COSINE_100_SUCC,
                    sim_measure < self.threshold_100,
                )

                data.add_property(
                    CosineSuccessRateProperties.COSINE_1000_SUCC,
                    sim_measure < self.threshold_1000,
                )
            else:
                # Impersonation
                data.add_property(
                    CosineSuccessRateProperties.COSINE_100_SUCC,
                    sim_measure > self.threshold_100,
                )

                data.add_property(
                    CosineSuccessRateProperties.COSINE_1000_SUCC,
                    sim_measure > self.threshold_1000,
                )

    def _collate_statistics(self, data_bin: DataBin) -> Any:
        results = {}

        cosine_100_succ = sum(
            data_bin.get_property(CosineSuccessRateProperties.COSINE_100_SUCC)
        )
        results["cosine_100_succ"] = cosine_100_succ
        results["cosine_100_succ_rate"] = cosine_100_succ / len(data_bin)

        cosine_1000_succ = sum(
            data_bin.get_property(CosineSuccessRateProperties.COSINE_1000_SUCC)
        )
        results["cosine_1000_succ"] = cosine_1000_succ
        results["cosine_1000_succ_rate"] = cosine_1000_succ / len(data_bin)

        return results

    def collate_statistics(self, data_bin: DataBin[BenchmarkData]) -> Any:
        # If we are collating results for one benchmark and one accessory, do not produce the max, min, mean etc. values
        if (
            len(data_bin.get_property(BenchmarkProperties.BENCHMARK)) == 1
            and len(data_bin.get_property(BenchmarkProperties.ACCESSORY)) == 1
        ):
            return self._collate_statistics(data_bin)

        grouped_bins = construct_bins_from_properties(
            data_bin, (BenchmarkProperties.BENCHMARK, BenchmarkProperties.ACCESSORY)
        )

        statistics = {}

        for group_bin in grouped_bins:
            group_statistics = self._collate_statistics(group_bin)
            for prop, value in group_statistics.items():
                values = statistics.get(prop, [])
                values.append(value)
                statistics[prop] = values

        results = {}

        for prop, values in statistics.items():
            data = np.array(values)

            if data.size == 0:
                results[prop] = {}
            elif data.size == 1:
                results[prop] = values[0]
            else:
                results[prop] = {
                    "mean": np.mean(data),
                    "std": np.std(data),
                    "max": np.max(data),
                    "min": np.min(data),
                }

        return results

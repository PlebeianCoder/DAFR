__all__ = ["SuccessRate", "SuccessRateProperties"]

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

device = torch.device("cuda")


class EmbedSuccessRateProperties(DataPropertyEnum):
    IMPERSONATED_F1 = "impersonated_f1"
    IMPERSONATED_F2 = "impersonated_f2"
    IMPERSONATED_F3 = "impersonated_f3"
    IMPERSONATED_F4 = "impersonated_f4"
    IMPERSONATED_F5 = "impersonated_f5"
    IMPERSONATED_F6 = "impersonated_f6"
    IMPERSONATED_F7 = "impersonated_f7"
    IMPERSONATED_F8 = "impersonated_f8"
    IMPERSONATED_F9 = "impersonated_f9"
    IMPERSONATED_F10 = "impersonated_f10"
    IMPERSONATED_OVERALL = "impersonated_overall"


def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output


def loadForLDM(imName, m):
    with torch.no_grad():
        if isinstance(imName, str):
            # String
            pilImg = Image.open(imName)
            img = np.asarray(pilImg).copy().astype(np.float32)
        elif isinstance(imName, Image.Image):
            # Pillow image
            pilImg = imName.copy()
            img = np.asarray(imName).copy().astype(np.float32)
        else:
            # Numpy array
            # Make channel go last
            if imName.shape[0] == 3:
                imName = np.transpose(imName, (1, 2, 0))
            img = imName.copy().astype(np.float32)
            pilImg = Image.fromarray(imName.astype(np.uint8))

        img = img.astype(np.float32) / 255.0  # Normalize
        if len(img.shape) != 3:
            return None, None  # To handle non colour images
        img = np.transpose(img, (2, 0, 1))  # Make channel go first
        img = torch.from_numpy(img).to(m)  # convert to torch
        return img, pilImg  # returns RGB


class EmbedSuccessRate(Statistic):
    class Factory(StatisticFactory):
        def __init__(
            self,
        ):
            pass

        @staticmethod
        def name() -> str:
            return "EmbedSuccessRate"

        def construct(self, benchmark_arguments, accessory) -> "EmbedSuccessRate":
            return EmbedSuccessRate(self.name(), benchmark_arguments, accessory)

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
            torchLogit = l2_norm(torch.from_numpy(data.augmented_logits).to(device))
            faceClasses = self._benchmark_arguments.size.class_names
            if not hasattr(self, "anchor"):
                wd = str(self._benchmark_arguments.weights_directory)
                rNetstr = "r18"
                if "r18" in wd:
                    if self._benchmark_arguments.size.is_large:
                        # then large
                        if self._benchmark_arguments.dataset == FaceDatasets.VGGFACE2:
                            # self.threshold =  [1.61, 1.61, 1.56, 1.56, 1.56, 1.63, 1.63, 1.63, 1.63, 1.62]
                            self.threshold = [
                                1.591,
                                1.565,
                                1.562,
                                1.571,
                                1.571,
                                1.625,
                                1.625,
                                1.627,
                                1.625,
                                1.625,
                            ]
                        else:
                            self.threshold = [1.8128, 1.68]
                    else:
                        if self._benchmark_arguments.dataset == FaceDatasets.VGGFACE2:
                            self.threshold = 1.7115
                        else:
                            # TO BE DEFINED
                            self.threshold = 1.5160

                elif "r34" in wd:
                    rNetstr = "r34"
                    if self._benchmark_arguments.size.is_large:
                        # then large
                        if self._benchmark_arguments.dataset == FaceDatasets.VGGFACE2:
                            self.threshold = [
                                1.589,
                                1.589,
                                1.589,
                                1.589,
                                1.589,
                                1.589,
                                1.636,
                                1.636,
                                1.589,
                                1.637,
                            ]
                        else:
                            self.threshold = [1.8503, 1.55]
                    else:
                        if self._benchmark_arguments.dataset == FaceDatasets.VGGFACE2:
                            self.threshold = 1.7438
                        else:
                            # TO BE DEFINED
                            self.threshold = 1.414
                elif "r50" in wd:
                    rNetstr = "r50"
                    if self._benchmark_arguments.size.is_large:
                        # then large
                        if self._benchmark_arguments.dataset == FaceDatasets.VGGFACE2:
                            self.threshold = [
                                1.585,
                                1.587,
                                1.587,
                                1.587,
                                1.587,
                                1.587,
                                1.635,
                                1.647,
                                1.63,
                                1.633,
                            ]
                        else:
                            self.threshold = [1.8575, 1.6]
                    else:
                        if self._benchmark_arguments.dataset == FaceDatasets.VGGFACE2:
                            self.threshold = 1.7605
                        else:
                            # TO BE DEFINED
                            self.threshold = 1.409

                datasetStr = "PUBFIG"
                if self._benchmark_arguments.dataset == FaceDatasets.VGGFACE2:
                    datasetStr = "VGGFACE2"
                self.anchor = torch.load(
                    f"../../../../anchors/{rNetstr}_{datasetStr}_{self._accessory.target_class}.pth",
                    map_location=device,
                )

            diff = torch.sum(
                torch.square(torch.subtract(torchLogit, self.anchor))
            ).item()

            succ = []
            for i in range(10):
                succ.append(diff <= self.threshold[i])
            succ_count = sum(succ)

            data.add_property(EmbedSuccessRateProperties.IMPERSONATED_F1, succ[0])
            data.add_property(EmbedSuccessRateProperties.IMPERSONATED_F2, succ[1])
            data.add_property(EmbedSuccessRateProperties.IMPERSONATED_F3, succ[2])
            data.add_property(EmbedSuccessRateProperties.IMPERSONATED_F4, succ[3])
            data.add_property(EmbedSuccessRateProperties.IMPERSONATED_F5, succ[4])
            data.add_property(EmbedSuccessRateProperties.IMPERSONATED_F6, succ[5])
            data.add_property(EmbedSuccessRateProperties.IMPERSONATED_F7, succ[6])
            data.add_property(EmbedSuccessRateProperties.IMPERSONATED_F8, succ[7])
            data.add_property(EmbedSuccessRateProperties.IMPERSONATED_F9, succ[8])
            data.add_property(EmbedSuccessRateProperties.IMPERSONATED_F10, succ[9])
            data.add_property(
                EmbedSuccessRateProperties.IMPERSONATED_OVERALL, succ_count
            )

    def _collate_statistics(self, data_bin: DataBin) -> Any:
        results = {}
        impersonated_f1 = sum(
            data_bin.get_property(EmbedSuccessRateProperties.IMPERSONATED_F1)
        )
        results["impersonated_f1"] = impersonated_f1
        results["impersonation_f1_success_rate"] = impersonated_f1 / len(data_bin)

        impersonated_f2 = sum(
            data_bin.get_property(EmbedSuccessRateProperties.IMPERSONATED_F2)
        )
        results["impersonated_f2"] = impersonated_f2
        results["impersonation_f2_success_rate"] = impersonated_f2 / len(data_bin)

        impersonated_f3 = sum(
            data_bin.get_property(EmbedSuccessRateProperties.IMPERSONATED_F3)
        )
        results["impersonated_f3"] = impersonated_f3
        results["impersonation_f3_success_rate"] = impersonated_f3 / len(data_bin)

        impersonated_f4 = sum(
            data_bin.get_property(EmbedSuccessRateProperties.IMPERSONATED_F4)
        )
        results["impersonated_f4"] = impersonated_f4
        results["impersonation_f4_success_rate"] = impersonated_f4 / len(data_bin)

        impersonated_f5 = sum(
            data_bin.get_property(EmbedSuccessRateProperties.IMPERSONATED_F5)
        )
        results["impersonated_f5"] = impersonated_f5
        results["impersonation_f5_success_rate"] = impersonated_f5 / len(data_bin)

        impersonated_f6 = sum(
            data_bin.get_property(EmbedSuccessRateProperties.IMPERSONATED_F1)
        )
        results["impersonated_f6"] = impersonated_f6
        results["impersonation_f6_success_rate"] = impersonated_f6 / len(data_bin)

        impersonated_f7 = sum(
            data_bin.get_property(EmbedSuccessRateProperties.IMPERSONATED_F7)
        )
        results["impersonated_f7"] = impersonated_f7
        results["impersonation_f7_success_rate"] = impersonated_f7 / len(data_bin)

        impersonated_f8 = sum(
            data_bin.get_property(EmbedSuccessRateProperties.IMPERSONATED_F8)
        )
        results["impersonated_f8"] = impersonated_f8
        results["impersonation_f8_success_rate"] = impersonated_f8 / len(data_bin)

        impersonated_f9 = sum(
            data_bin.get_property(EmbedSuccessRateProperties.IMPERSONATED_F9)
        )
        results["impersonated_f9"] = impersonated_f9
        results["impersonation_f9_success_rate"] = impersonated_f9 / len(data_bin)

        impersonated_f10 = sum(
            data_bin.get_property(EmbedSuccessRateProperties.IMPERSONATED_F10)
        )
        results["impersonated_f10"] = impersonated_f10
        results["impersonation_f10_success_rate"] = impersonated_f10 / len(data_bin)

        impersonated_OVERALL = sum(
            data_bin.get_property(EmbedSuccessRateProperties.IMPERSONATED_OVERALL)
        )
        results["impersonated_OVERALL"] = impersonated_OVERALL
        results["impersonation_OVERALL_success_rate"] = impersonated_OVERALL / (
            len(data_bin) * 10
        )

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

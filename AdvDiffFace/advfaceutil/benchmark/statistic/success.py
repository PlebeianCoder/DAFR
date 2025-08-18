__all__ = ["SuccessRate", "SuccessRateProperties"]

from typing import Any

import numpy as np

from advfaceutil.benchmark.factory import default_factory
from advfaceutil.benchmark.data import (
    BenchmarkData,
    BenchmarkProperties,
    DataPropertyEnum,
    DataBin,
    construct_bins_from_properties,
)
from advfaceutil.benchmark.statistic.base import Statistic


class SuccessRateProperties(DataPropertyEnum):
    SUCCESSFULLY_CLASSIFIED = "successfully_classified"
    DODGED = "dodged"
    IMPERSONATED = "impersonated"


@default_factory("SuccessRate")
class SuccessRate(Statistic):
    def record_statistic(self, data: BenchmarkData) -> None:
        data.add_property(
            SuccessRateProperties.SUCCESSFULLY_CLASSIFIED,
            data.predicted_class_index == data.class_index,
        )

        if self._accessory.is_impersonation:
            data.add_property(
                SuccessRateProperties.IMPERSONATED,
                data.augmented_predicted_class_index
                == self._accessory.target_class_index,
            )

        # Record as being dodged if the augmented predicted index is not the actual class image and we are not ignoring
        # this data (i.e. the class is the target class)
        data.add_property(
            SuccessRateProperties.DODGED,
            data.augmented_predicted_class_index != data.class_index
            and data.class_index != self._accessory.target_class_index,
        )

    def _collate_statistics(self, data_bin: DataBin) -> Any:
        success = sum(
            data_bin.get_property(SuccessRateProperties.SUCCESSFULLY_CLASSIFIED)
        )

        results = {
            "baseline_successes": success,
            "baseline_success_rate": success / len(data_bin),
        }

        # If the class of the image is the target class then we don't record the dodge statistic
        dodges_to_ignore = len(
            [
                data
                for data in data_bin
                if data.class_index == self._accessory.target_class
            ]
        )

        # If we have not ignored all dodges then calculate the dodge rate
        if len(data_bin) != dodges_to_ignore:
            dodges = sum(data_bin.get_property(SuccessRateProperties.DODGED))
            results["dodges"] = dodges
            results["dodge_success_rate"] = dodges / (len(data_bin) - dodges_to_ignore)

        if self._accessory.is_impersonation:
            impersonations = sum(
                data_bin.get_property(SuccessRateProperties.IMPERSONATED)
            )
            results["impersonations"] = impersonations
            results["impersonation_success_rate"] = impersonations / len(data_bin)

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

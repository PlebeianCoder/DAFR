__all__ = ["UniversalClasses", "UniversalClassesProperties"]

from typing import Generator
from functools import partial

from advfaceutil.benchmark.augmentation.base import Augmentation
from advfaceutil.benchmark.factory import default_factory
from advfaceutil.benchmark.data import (
    BenchmarkData,
    DataPropertyEnum,
    load_data,
    data_count,
)


class UniversalClassesProperties(DataPropertyEnum):
    UNIVERSAL_EXAMPLE = "universal_example"
    CLASS_NAME = "class_name"


@default_factory("UniversalClasses")
class UniversalClasses(Augmentation):
    def _label_data(self, data: BenchmarkData):
        data.add_property(
            UniversalClassesProperties.UNIVERSAL_EXAMPLE,
            data.class_index != self._accessory.base_class_index,
        )
        data.add_property(UniversalClassesProperties.CLASS_NAME, data.class_name)

    def load_extra_data(self) -> Generator[BenchmarkData, None, None]:
        label = partial(self._label_data)
        if self._accessory.universal_class_indices is not None:
            for i in self._accessory.universal_class_indices:
                if i != self._accessory.base_class_index:
                    class_name = self._benchmark_arguments.size.class_names[i]
                    if class_name in self._benchmark_arguments.size.dataset_names:
                        yield from load_data(
                            self._benchmark_arguments.dataset,
                            self._benchmark_arguments.dataset_directory,
                            class_name,
                            i,
                            self._benchmark_arguments.class_image_limit,
                            label,
                        )
                    else:
                        yield from load_data(
                            self._benchmark_arguments.dataset,
                            self._benchmark_arguments.researchers_directory,
                            class_name,
                            i,
                            self._benchmark_arguments.class_image_limit,
                            label,
                        )
        else:
            for i, class_name in enumerate(self._benchmark_arguments.size.class_names):
                if i == self._accessory.base_class_index:
                    continue
                if class_name in self._benchmark_arguments.size.dataset_names:
                    yield from load_data(
                        self._benchmark_arguments.dataset,
                        self._benchmark_arguments.dataset_directory,
                        class_name,
                        i,
                        self._benchmark_arguments.class_image_limit,
                        label,
                    )
                else:
                    yield from load_data(
                        self._benchmark_arguments.dataset,
                        self._benchmark_arguments.researchers_directory,
                        class_name,
                        i,
                        self._benchmark_arguments.class_image_limit,
                        label,
                    )

    def extra_data_count(self) -> int:
        if self._accessory.universal_class_indices is not None:
            return sum(
                data_count(
                    self._benchmark_arguments.dataset,
                    self._benchmark_arguments.dataset_directory,
                    self._benchmark_arguments.size.class_names[i],
                    self._benchmark_arguments.class_image_limit,
                )
                for i in self._accessory.universal_class_indices
                if i != self._accessory.base_class_index
            ) + sum(
                data_count(
                    self._benchmark_arguments.dataset,
                    self._benchmark_arguments.researchers_directory,
                    self._benchmark_arguments.size.class_names[i],
                    self._benchmark_arguments.class_image_limit,
                )
                for i in self._accessory.universal_class_indices
                if self._benchmark_arguments.size.class_names[i]
                not in self._benchmark_arguments.size.dataset_names
            )
        return sum(
            data_count(
                self._benchmark_arguments.dataset,
                self._benchmark_arguments.dataset_directory,
                class_name,
                self._benchmark_arguments.class_image_limit,
            )
            for i, class_name in enumerate(self._benchmark_arguments.size.class_names)
            if i != self._accessory.base_class_index
            and class_name in self._benchmark_arguments.size.dataset_names
        ) + sum(
            data_count(
                self._benchmark_arguments.dataset,
                self._benchmark_arguments.researchers_directory,
                class_name,
                self._benchmark_arguments.class_image_limit,
            )
            for i, class_name in enumerate(self._benchmark_arguments.size.class_names)
            if i != self._accessory.base_class_index
            and class_name not in self._benchmark_arguments.size.dataset_names
        )

    def pre_augmentation_processing(self, data: BenchmarkData) -> BenchmarkData:
        # Add missing properties to the data. These will be missing for the non-universal examples
        if not data.has_property(UniversalClassesProperties.UNIVERSAL_EXAMPLE):
            data.add_property(UniversalClassesProperties.UNIVERSAL_EXAMPLE, False)
        if not data.has_property(UniversalClassesProperties.CLASS_NAME):
            data.add_property(UniversalClassesProperties.CLASS_NAME, data.class_name)

        return data

__all__ = ["Benchmark", "BenchmarkFactory"]

from abc import ABCMeta
from abc import abstractmethod
from itertools import chain
from logging import getLogger, Logger
from time import time
from typing import Generator, List, Tuple
from typing import Optional

import cv2
import numpy as np
import torch
from pyfacear import OBJMeshIO

from advfaceutil.benchmark.args import BenchmarkArguments
from advfaceutil.benchmark.augmentation import Augmentation
from advfaceutil.benchmark.augmentation import AugmentationFactory
from advfaceutil.benchmark.data import BenchmarkProperties
from advfaceutil.benchmark.data import BenchmarkData
from advfaceutil.benchmark.data import Accessory
from advfaceutil.benchmark.data import CompressedBenchmarkData
from advfaceutil.benchmark.data import DataBin
from advfaceutil.benchmark.data import construct_bins_from_properties
from advfaceutil.benchmark.data import data_count
from advfaceutil.benchmark.data import DataProperty
from advfaceutil.benchmark.data import get_bin_file_name
from advfaceutil.benchmark.data import load_data
from advfaceutil.benchmark.factory import construct_all
from advfaceutil.benchmark.factory import Factory
from advfaceutil.benchmark.factory import FactoryInstance
from advfaceutil.benchmark.statistic import AccessoryStatistic
from advfaceutil.benchmark.statistic import collate_statistics_for_bin
from advfaceutil.benchmark.statistic import Statistic
from advfaceutil.benchmark.statistic import StatisticFactory
from advfaceutil.recognition import RecognitionArchitecture
from advfaceutil.recognition.processing import AugmentationOptions
from advfaceutil.recognition.processing import DlibAugmentationOptions
from advfaceutil.recognition.processing import FaceProcessor
from advfaceutil.recognition.processing import FaceProcessors
from advfaceutil.recognition.processing import MediaPipeAugmentationOptions
from advfaceutil.recognition.processing import AdvMaskAugmentationOptions
from advfaceutil.utils import to_pretty_json


class BenchmarkFactory(Factory, metaclass=ABCMeta):
    @abstractmethod
    def construct(
        self,
        benchmark_arguments: BenchmarkArguments,
        accessory: Accessory,
    ) -> "Benchmark":
        pass


class Benchmark(FactoryInstance):
    class Factory(BenchmarkFactory, metaclass=ABCMeta):
        def __init__(
            self,
            augmentations: List[AugmentationFactory],
            statistics: List[StatisticFactory],
            accessory_statistics: List[AccessoryStatistic],
            bin_property_combinations: List[Tuple[DataProperty, ...]],
        ):
            self.augmentations = augmentations
            self.statistics = statistics
            self.accessory_statistics = accessory_statistics
            self.bin_property_combinations = bin_property_combinations

        def construct(
            self,
            benchmark_arguments: BenchmarkArguments,
            accessory: Accessory,
        ) -> "Benchmark":
            return Benchmark(
                self.name(),
                benchmark_arguments,
                accessory,
                construct_all(self.augmentations, benchmark_arguments, accessory),
                construct_all(self.statistics, benchmark_arguments, accessory),
                self.accessory_statistics,
                self.bin_property_combinations,
            )

    def __init__(
        self,
        name: str,
        benchmark_arguments: BenchmarkArguments,
        accessory: Accessory,
        augmentations: List[Augmentation],
        statistics: List[Statistic],
        accessory_statistics: List[AccessoryStatistic],
        bin_property_combinations: List[Tuple[DataProperty, ...]],
    ):
        super().__init__(name, benchmark_arguments, accessory)
        self._augmentations = augmentations
        self._statistics = statistics
        self._accessory_statistics = accessory_statistics
        self._bin_property_combinations = bin_property_combinations

        self._output_directory = (
            benchmark_arguments.output_directory
            / "benchmarks"
            / accessory.adversarial_accessory_path.stem
            / name
        )
        self._output_directory.mkdir(parents=True, exist_ok=True)

        self._image_directory = self._output_directory / "images"
        # Only create the image output directory if we are saving images
        if (
            benchmark_arguments.save_augmented_images
            or benchmark_arguments.save_aligned_images
        ):
            self._image_directory.mkdir(parents=True, exist_ok=True)

        self._saved_image_index = {}

        self.__logger = getLogger(f"benchmark/{self.name}")

        # Add the benchmark property to the accessory
        self._accessory.add_property(BenchmarkProperties.BENCHMARK, self.name)

    @property
    def logger(self) -> Logger:
        return self.__logger

    def __repr__(self) -> str:
        return f"Benchmark(name={self.name})"

    @property
    def benchmark_arguments(self) -> BenchmarkArguments:
        return self._benchmark_arguments

    @property
    def accessory(self) -> Accessory:
        return self._accessory

    @property
    def statistics(self) -> List[Statistic]:
        return self._statistics

    @property
    def accessory_statistics(self) -> List[AccessoryStatistic]:
        return self._accessory_statistics

    @property
    def bin_property_combinations(self) -> List[Tuple[DataProperty, ...]]:
        return self._bin_property_combinations

    def _prepare_data_loaders(
        self,
    ) -> Tuple[List[Generator[BenchmarkData, None, None]], int]:
        self.__logger.info("Preparing data loaders")

        # Create the data generators for the base class depending on whether the base class is within the dataset
        # or the researchers directory
        if self._accessory.base_class in self._benchmark_arguments.size.dataset_names:
            data_loaders = [
                load_data(
                    self._benchmark_arguments.dataset,
                    self._benchmark_arguments.dataset_directory,
                    self._accessory.base_class,
                    self._accessory.base_class_index,
                    self._benchmark_arguments.class_image_limit,
                )
            ]
            expected_data_count = data_count(
                self._benchmark_arguments.dataset,
                self._benchmark_arguments.dataset_directory,
                self._accessory.base_class,
                self._benchmark_arguments.class_image_limit,
            )
        else:
            data_loaders = [
                load_data(
                    self._benchmark_arguments.dataset,
                    self._benchmark_arguments.researchers_directory,
                    self._accessory.base_class,
                    self._accessory.base_class_index,
                    self._benchmark_arguments.class_image_limit,
                )
            ]
            expected_data_count = data_count(
                self._benchmark_arguments.dataset,
                self._benchmark_arguments.researchers_directory,
                self._accessory.base_class,
                self._benchmark_arguments.class_image_limit,
            )

        # Load any data from augmentation (e.g., data flipping, universal classes etc.)
        for augmentation in self._augmentations:
            generator = augmentation.load_extra_data()
            if generator is not None:
                data_loaders.append(generator)
                # Increase the expected amount of data
                expected_data_count += augmentation.extra_data_count()

        # Add the expected amount of data once we have loaded all the data
        # (i.e., data that are derived from loaded data)
        derived_image_count = 0
        for augmentation in self._augmentations:
            derived_image_count += augmentation.extra_data_count_from_loaded_data(
                expected_data_count
            )

        expected_data_count += derived_image_count

        return data_loaders, expected_data_count

    def _derive_data(self, data: BenchmarkData) -> List[BenchmarkData]:
        derived_data = [data]
        for augmentation in self._augmentations:
            extra_images = augmentation.load_extra_data_from_data(data)
            if extra_images:
                derived_data.extend(extra_images)

        return derived_data

    def _pre_augmentation_processing(self, data: BenchmarkData) -> BenchmarkData:
        for augmentation in self._augmentations:
            data = augmentation.pre_augmentation_processing(data)
        return data

    def _post_augmentation_processing(self, data: BenchmarkData) -> BenchmarkData:
        for augmentation in self._augmentations:
            data = augmentation.post_augmentation_processing(data)
        return data

    def _record_statistics(self, data: BenchmarkData) -> None:
        data.add_property(BenchmarkProperties.BENCHMARK, self.name)
        data.add_property(BenchmarkProperties.ACCESSORY, self._accessory)

        for statistic in self._statistics:
            statistic.record_statistic(data)

    def _save_image(self, image: np.ndarray, class_name: str, name: str) -> None:
        index = self._saved_image_index.get((class_name, name), 0)
        self._saved_image_index[(class_name, name)] = index + 1
        cv2.imwrite(
            (self._image_directory / f"{index}_{class_name}_{name}.png").as_posix(),
            cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
        )

    def _classify_data(
        self, data: BenchmarkData, architecture: RecognitionArchitecture
    ):
        # Classify the augmented image
        # logits = architecture.logits(data.augmented_aligned_image)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        pre_aug = architecture.preprocess(
            data.augmented_aligned_image, toBGR=False, batched=True
        ).to(device)
        logits = architecture.returnEmbedding(pre_aug)
        # print(logits)
        # print(logits.argmax())
        # print(logits.size())
        # if logits.ndim >= 1:
        #     logits = torch.squeeze(logits, 0)
        data.augmented_logits = logits.detach().cpu().numpy()

        # Store the predicted class
        # data.augmented_predicted_class_index = logits.argmax().item()
        # data.augmented_predicted_class = self._benchmark_arguments.size.class_names[
        #     data.augmented_predicted_class_index
        # ]
        data.augmented_predicted_class_index = logits.argmax().item()
        data.augmented_predicted_class = ""

        # Classify the original image
        pre_aligned = architecture.preprocess(
            data.aligned_image, toBGR=False, batched=True
        ).to(device)
        logits = architecture.returnEmbedding(pre_aligned)
        # print(logits.size())
        # if logits.ndim = 1:
        #     logits = torch.squeeze(logits, 0)
        data.logits = logits.detach().cpu().numpy()

        # Store the predicted class
        # data.predicted_class_index = logits.argmax().item()
        # data.predicted_class = self._benchmark_arguments.size.class_names[
        #     data.predicted_class_index
        # ]
        data.predicted_class_index = logits.argmax().item()
        data.predicted_class = ""

    def _process_data(
        self,
        data: BenchmarkData,
        face_processor: FaceProcessor,
        augmentation_options: AugmentationOptions,
        architecture: RecognitionArchitecture,
    ) -> Optional[CompressedBenchmarkData]:
        # Find the faces in the data
        faces = face_processor.detect_faces(data.image)

        # Limit to exactly 1 face
        if faces is None or len(faces) != 1:
            return None

        face = faces[0]

        # Pre-augmentation processing
        data = self._pre_augmentation_processing(data)

        # Augment the image
        data.augmented_image = face_processor.augment(
            data.image, augmentation_options, face
        )

        # Post augmentation processing
        data = self._post_augmentation_processing(data)

        # Save the augmented image
        if self._benchmark_arguments.save_augmented_images:
            self._save_image(data.augmented_image, data.class_name, "augmented")

        # Align the image
        data.augmented_aligned_image = face_processor.align(
            data.augmented_image, architecture.crop_size, face
        )

        # Save the aligned image
        if self._benchmark_arguments.save_aligned_images:
            self._save_image(data.augmented_aligned_image, data.class_name, "aligned")

        # Align the original image
        data.aligned_image = face_processor.align(
            data.image, architecture.crop_size, face
        )

        # Classify the image
        self._classify_data(data, architecture)

        # Record the statistics
        self._record_statistics(data)

        # Add any properties that are on the accessory to the benchmark data
        # Note: this occurs before accessory statistics have been recorded
        # This is useful for example for grouping results based on accessory types if we test various accessory types
        # within a benchmark suite.
        data.copy_from(self._accessory)

        return data.compress()

    def _collate_statistics(self, data: List[CompressedBenchmarkData]):
        statistics_directory = self._output_directory / "statistics"
        statistics_directory.mkdir(parents=True, exist_ok=True)

        # Record and store the accessory statistics
        for accessory_statistic in self._accessory_statistics:
            accessory_statistic.record_statistic(self._accessory)

        accessory_statistics = collate_statistics_for_bin(
            self._accessory_statistics, DataBin((), [self._accessory])
        )

        (statistics_directory / "accessory.json").write_text(
            to_pretty_json(accessory_statistics)
        )

        for properties in self._bin_property_combinations:
            bins = construct_bins_from_properties(data, properties)
            for data_bin in bins:
                statistics = collate_statistics_for_bin(self._statistics, data_bin)

                file = statistics_directory / (
                    get_bin_file_name(data_bin, properties) + ".json"
                )
                file.write_text(to_pretty_json(statistics))

        data_bin = DataBin((), data)
        statistics = collate_statistics_for_bin(self._statistics, data_bin)

        file = statistics_directory / "all.json"
        file.write_text(to_pretty_json(statistics))

        self.__logger.info("Saved statistics to %s", statistics_directory)

    def _save_config(self):
        config_directory = self._output_directory / "config"
        config_directory.mkdir(parents=True, exist_ok=True)

        (config_directory / "benchmark.json").write_text(
            to_pretty_json(self._benchmark_arguments)
        )
        (config_directory / "accessory.json").write_text(
            to_pretty_json(self._accessory)
        )
        (config_directory / "augmentations.json").write_text(
            to_pretty_json(self._augmentations)
        )
        (config_directory / "statistics.json").write_text(
            to_pretty_json(self._statistics)
        )

        self.__logger.info(
            "Configuration settings saved to %s", config_directory.as_posix()
        )

    def run(self) -> List[CompressedBenchmarkData]:
        start_time = time()

        self.__logger.info(
            "Beginning '%s' benchmark for accessory '%s' for base class '%s' targeting '%s'",
            self.name,
            self._accessory.adversarial_accessory_path,
            self._accessory.base_class,
            self._accessory.target_class,
        )

        # Load the dataset and researchers
        data_loaders, expected_data_count = self._prepare_data_loaders()

        self.__logger.info("Expecting to process %d items of data", expected_data_count)

        dataset = chain(*data_loaders)

        self.__logger.info("Constructing models")

        # Load the recognition model
        architecture = self._benchmark_arguments.architecture.construct(
            self._benchmark_arguments.dataset,
            self._benchmark_arguments.size,
            self._benchmark_arguments.weights_directory,
        )

        # Load the face processor and augmentation options
        face_processor = self._benchmark_arguments.face_processor.construct()

        # If using dlib, use the dlib options
        if self._benchmark_arguments.face_processor == FaceProcessors.DLIB:
            augmentation_options = DlibAugmentationOptions(
                texture=self._accessory.adversarial_accessory,
                additive_overlay=self._benchmark_arguments.additive_overlay,
            )
        elif self._benchmark_arguments.face_processor == FaceProcessors.ADV:
            augmentation_options = AdvMaskAugmentationOptions(
                self._accessory.adversarial_accessory
            )
        else:
            # If using mediapipe, load the overlay model if given
            model = None

            if self._benchmark_arguments.overlay_model:
                model = OBJMeshIO.load(self._benchmark_arguments.overlay_model)

            augmentation_options = MediaPipeAugmentationOptions(
                texture=self._accessory.adversarial_accessory,
                mesh=model,
            )

        total = 0

        self.__logger.info("Starting benchmarking loop")

        current_time = time()
        average_time = None

        compressed_dataset = []

        # For each image from the loaders:
        for base_data in dataset:
            derived_data = self._derive_data(base_data)

            for data in derived_data:
                compressed_data = self._process_data(
                    data, face_processor, augmentation_options, architecture
                )

                if compressed_data is not None:
                    compressed_dataset.append(compressed_data)

                total += 1

                if total % 100 == 0:
                    new_time = time()

                    time_difference = new_time - current_time

                    current_time = new_time

                    if average_time is None:
                        average_time = time_difference
                    else:
                        average_time = (average_time + time_difference) / 2

                    estimated_time = average_time / 100 * (expected_data_count - total)

                    if estimated_time > 60:
                        self.__logger.info(
                            f"Processed {total}/{expected_data_count} dataset. "
                            f"{total / expected_data_count * 100:.2f}% complete. "
                            f"Estimated time remaining: {int(estimated_time // 60)}m {estimated_time % 60:.2f}s"
                        )
                    else:
                        self.__logger.info(
                            f"Processed {total}/{expected_data_count} dataset. "
                            f"{total / expected_data_count * 100:.2f}% complete. "
                            f"Estimated time remaining: {estimated_time:.2f}s"
                        )

        self._collate_statistics(compressed_dataset)

        end_time = time()

        self.__logger.info(
            "'%s' benchmark finished for accessory '%s' for base class '%s' targeting '%s'",
            self.name,
            self._accessory.adversarial_accessory_path,
            self._accessory.base_class,
            self._accessory.target_class,
        )
        self.__logger.info(f"Total time: {end_time - start_time:.2f}s")

        self._save_config()

        if self._benchmark_arguments.save_raw_statistics:
            (self._benchmark_arguments.output_directory / "raw.json").write_text(
                to_pretty_json(
                    list(map(lambda d: d.to_json_dict(), compressed_dataset))
                )
            )

        return compressed_dataset

__all__ = ["BenchmarkSuite", "BenchmarkResult"]

from concurrent.futures import ProcessPoolExecutor, as_completed, Future
from dataclasses import dataclass, field
from logging import getLogger
from multiprocessing import Manager
from multiprocessing import cpu_count
from multiprocessing import log_to_stderr as mp_log_to_stderr
from pathlib import Path
from time import time
from typing import List, Set, Tuple, Optional

import torch.multiprocessing

from advfaceutil.benchmark.args import BenchmarkArguments
from advfaceutil.benchmark.base import Benchmark
from advfaceutil.benchmark.base import BenchmarkFactory
from advfaceutil.benchmark.data import CompressedBenchmarkData, Accessory
from advfaceutil.benchmark.data import DataBin
from advfaceutil.benchmark.data import DataProperty
from advfaceutil.benchmark.data import construct_bins_from_properties
from advfaceutil.benchmark.data import get_bin_file_name
from advfaceutil.benchmark.gpu import GPUDeviceManager
from advfaceutil.benchmark.statistic import AccessoryStatistic
from advfaceutil.benchmark.statistic import Statistic
from advfaceutil.benchmark.statistic import collate_statistics_for_bin
from advfaceutil.benchmark.suppliers import AccessoriesSupplier
from advfaceutil.utils import (
    to_pretty_json,
    LoggingThread,
    configure_loggers_on_worker,
)

LOGGER = getLogger("benchmark_suite")


@dataclass(frozen=True)
class BenchmarkResult:
    benchmark: Benchmark
    data: List[CompressedBenchmarkData] = field(repr=False)


class BenchmarkSuite:
    def __init__(
        self,
        benchmarks: List[BenchmarkFactory],
        accessories_suppliers: List[AccessoriesSupplier],
        accessory_bin_property_combinations: Optional[
            List[Tuple[DataProperty, ...]]
        ] = None,
    ):
        self._benchmarks = benchmarks
        self._accessories_suppliers = accessories_suppliers

        if accessory_bin_property_combinations is None:
            accessory_bin_property_combinations = []

        self._accessory_bin_property_combinations = accessory_bin_property_combinations

    @staticmethod
    def _get_common_statistics(results: List[BenchmarkResult]) -> Set[Statistic]:
        if len(results) == 0:
            return set()

        common_statistics = set(results[0].benchmark.statistics)
        for result in results[1:]:
            common_statistics.intersection_update(result.benchmark.statistics)

        return common_statistics

    @staticmethod
    def _get_common_accessory_statistics(
        results: List[BenchmarkResult],
    ) -> Set[AccessoryStatistic]:
        if len(results) == 0:
            return set()

        common_accessory_statistics = set(results[0].benchmark.accessory_statistics)
        for result in results[1:]:
            common_accessory_statistics.intersection_update(
                result.benchmark.accessory_statistics
            )

        return common_accessory_statistics

    @staticmethod
    def _get_common_bin_property_combinations(
        results: List[BenchmarkResult],
    ) -> Set[Tuple[DataProperty, ...]]:
        if len(results) == 0:
            return set()

        common_bin_property_combinations = set(
            results[0].benchmark.bin_property_combinations
        )
        for result in results[1:]:
            common_bin_property_combinations.intersection_update(
                result.benchmark.bin_property_combinations
            )

        return common_bin_property_combinations

    @staticmethod
    def _collate_group_statistics(
        group: List[BenchmarkResult],
        statistics_directory: Path,
        common_statistics: Set[Statistic],
        common_bin_property_combinations: Set[Tuple[DataProperty, ...]],
    ):
        data = []
        for result in group:
            data.extend(result.data)

        for properties in common_bin_property_combinations:
            bins = construct_bins_from_properties(data, properties)
            for data_bin in bins:
                statistics = collate_statistics_for_bin(common_statistics, data_bin)

                file = statistics_directory / (
                    get_bin_file_name(data_bin, properties) + ".json"
                )
                file.write_text(to_pretty_json(statistics))

        data_bin = DataBin((), data)
        statistics = collate_statistics_for_bin(common_statistics, data_bin)

        file = statistics_directory / "all.json"
        file.write_text(to_pretty_json(statistics))

    def _collate_accessories_statistics(
        self,
        accessories: List[Accessory],
        statistics_directory: Path,
        common_statistics: Set[AccessoryStatistic],
    ):
        if len(accessories) == 1 or len(self._accessory_bin_property_combinations) == 0:
            accessory_statistics = collate_statistics_for_bin(
                common_statistics, DataBin((), accessories)
            )

            (statistics_directory / "accessory.json").write_text(
                to_pretty_json(accessory_statistics)
            )
        else:
            accessories_statistic_directory = statistics_directory / "accessories"
            accessories_statistic_directory.mkdir(parents=True, exist_ok=True)

            for properties in self._accessory_bin_property_combinations:
                bins = construct_bins_from_properties(accessories, properties)
                for data_bin in bins:
                    statistics = collate_statistics_for_bin(common_statistics, data_bin)

                    file = accessories_statistic_directory / (
                        get_bin_file_name(data_bin, properties) + ".json"
                    )
                    file.write_text(to_pretty_json(statistics))

            data_bin = DataBin((), accessories)
            statistics = collate_statistics_for_bin(common_statistics, data_bin)

            file = accessories_statistic_directory / "all.json"
            file.write_text(to_pretty_json(statistics))

    def _collate_statistics(
        self, results: List[BenchmarkResult], output_directory: Path
    ):
        if len(results) == 0:
            return

        LOGGER.info("Collating collective statistics")

        benchmark_directory = output_directory / "benchmarks"

        # Sort the results based on their accessory arguments
        results.sort(key=lambda r: r.benchmark.accessory)

        # Group the benchmarks based on those for the same accessory
        groups = [
            results[i : i + len(self._benchmarks)]
            for i in range(0, len(results), len(self._benchmarks))
        ]

        common_accessory_statistics = self._get_common_accessory_statistics(groups[0])
        common_statistics = self._get_common_statistics(groups[0])
        common_bin_property_combinations = self._get_common_bin_property_combinations(
            groups[0]
        )

        for group in groups:
            accessory = group[0].benchmark.accessory

            # Create the directory to store the statistics for the group
            statistics_directory = (
                benchmark_directory
                / accessory.adversarial_accessory_path.stem
                / "statistics"
            )
            statistics_directory.mkdir(parents=True, exist_ok=True)

            self._collate_accessories_statistics(
                [accessory], statistics_directory, common_accessory_statistics
            )

            self._collate_group_statistics(
                group,
                statistics_directory,
                common_statistics,
                common_bin_property_combinations,
            )

        root_statistics = output_directory / "statistics"

        # Group the benchmarks based on the type of benchmark
        groups = [
            results[i :: len(self._benchmarks)] for i in range(len(self._benchmarks))
        ]

        for group in groups:
            common_accessory_statistics = self._get_common_accessory_statistics(group)
            common_statistics = self._get_common_statistics(group)
            common_bin_property_combinations = (
                self._get_common_bin_property_combinations(group)
            )

            # Create the directory to store the statistics for the group
            statistics_directory = root_statistics / group[0].benchmark.name
            statistics_directory.mkdir(parents=True, exist_ok=True)

            self._collate_accessories_statistics(
                list(set(map(lambda r: r.benchmark.accessory, group))),
                statistics_directory,
                common_accessory_statistics,
            )

            self._collate_group_statistics(
                group,
                statistics_directory,
                common_statistics,
                common_bin_property_combinations,
            )

        # Combine all the data and statistics together
        common_accessory_statistics = self._get_common_accessory_statistics(results)
        common_statistics = self._get_common_statistics(results)
        common_bin_property_combinations = self._get_common_bin_property_combinations(
            results
        )

        # Create the directory to store the statistics for the group
        statistics_directory = root_statistics / "all"
        statistics_directory.mkdir(parents=True, exist_ok=True)

        self._collate_accessories_statistics(
            list(set(map(lambda r: r.benchmark.accessory, results))),
            statistics_directory,
            common_accessory_statistics,
        )

        self._collate_group_statistics(
            results,
            statistics_directory,
            common_statistics,
            common_bin_property_combinations,
        )

        LOGGER.info("Benchmark statistics collated and saved to '%s'", output_directory)

    @staticmethod
    def _process_benchmark(benchmark: Benchmark) -> BenchmarkResult:
        result = benchmark.run()
        benchmark_result = BenchmarkResult(benchmark, result)
        return benchmark_result

    @staticmethod
    def _generate_accessories(
        gpu_manager: GPUDeviceManager,
        supplier: AccessoriesSupplier,
    ) -> List[Accessory]:
        logger = getLogger(f"benchmark_suite/generation")
        logger.info("Generating accessory using generator %s", type(supplier).__name__)

        accessories = supplier.supply(gpu_manager)

        logger.info("Generated %d accessories", len(accessories))

        return accessories

    def save_checkpoint(
        self,
        benchmark_arguments: BenchmarkArguments,
        next_supplier_index: int,
        results: List[BenchmarkResult],
    ):
        from advfaceutil.benchmark.checkpointing import Checkpoint

        checkpoint = Checkpoint(
            self._benchmarks,
            benchmark_arguments,
            self._accessories_suppliers,
            next_supplier_index,
            results,
        )
        checkpoint.save()

    def _future_done_callback(
        self,
        benchmark_arguments: BenchmarkArguments,
        results: List[BenchmarkResult],
        next_supplier_index: int,
        future: Future,
    ) -> None:
        if future.cancelled():
            return

        # Copy the results so any changes we make are not reflected in the actual benchmarking
        results = results.copy()

        # If the current benchmark is not in the results then add it
        current_benchmark_result = future.result()
        if current_benchmark_result not in results:
            results.append(current_benchmark_result)

        # Get the accessory results for the current accessory
        current_accessory_results = list(
            filter(
                lambda r: r.benchmark.accessory
                == current_benchmark_result.benchmark.accessory,
                results,
            )
        )
        # If we do not have results for each benchmark for the current accessory then we should not create a checkpoint
        if len(current_accessory_results) != len(self._benchmarks):
            return

        # The results list may contain results of accessories that have not finished all benchmarks
        # In this case, those unfinished benchmarks should not be included in the checkpoint

        # Group the results together based on the accessory they belong to
        grouped_results = {}

        for result in results:
            group = grouped_results.get(result.benchmark.accessory, [])
            group.append(result)
            grouped_results[result.benchmark.accessory] = group

        # Remove any results that belong to accessories that have not finished benchmarking
        for group in grouped_results.values():
            # If the benchmarks for an accessory has not finished, remove all the benchmark results that belong to
            # the group
            if len(group) != len(self._benchmarks):
                results = [result for result in results if result not in group]

        self.save_checkpoint(benchmark_arguments, next_supplier_index, results)

    def run(
        self,
        benchmark_arguments: BenchmarkArguments,
        initial_results: Optional[List[BenchmarkResult]] = None,
    ):
        if initial_results is None:
            initial_results = []

        worker_count = (
            benchmark_arguments.worker_count + 1
            if benchmark_arguments.worker_count is not None
            else cpu_count()
        )

        LOGGER.info(
            "Starting benchmarking suite using %d processes and benchmarks: %s",
            worker_count,
            ", ".join(benchmark.name() for benchmark in self._benchmarks),
        )

        # Get the root logger and get the log level
        root = getLogger()
        log_level = root.level

        start_time = time()

        # Change the PyTorch multiprocessing to use Spawn instead of Fork
        torch.multiprocessing.set_start_method("spawn", force=True)

        # Use a multiprocessing manager to create the queue
        with Manager() as manager:
            log_queue = manager.Queue()

            logging_thread = LoggingThread(log_queue)

            # Initialise logging when using multiprocessing
            mp_log_to_stderr()

            # Create the GPU device manager
            gpu_manager = GPUDeviceManager(
                manager,
                processes_per_gpu=benchmark_arguments.accessory_generation_processes_per_gpu,
            )

            LOGGER.info(
                "Created the GPU Device Manager for %d GPUs with a maximum of %d processes per GPU",
                gpu_manager.device_count,
                gpu_manager.processes_per_gpu,
            )

            maximum_accessory_generation_workers_count = (
                gpu_manager.device_count * gpu_manager.processes_per_gpu
            )
            accessory_generation_workers_count = (
                benchmark_arguments.accessory_generation_workers_count
            )

            # Ensure that the number of accessory generation processes is at most the maximum
            if (
                accessory_generation_workers_count
                > maximum_accessory_generation_workers_count
            ):
                LOGGER.warning(
                    "Reducing the number of accessory generation workers from %d to %d to reflect the number"
                    "of GPU devices and to avoid unnecessary waiting on GPU availability.",
                    accessory_generation_workers_count,
                    maximum_accessory_generation_workers_count,
                )
                accessory_generation_workers_count = (
                    maximum_accessory_generation_workers_count
                )

            results = initial_results

            # Use a process pool to execute the benchmarks over multiple processes
            with ProcessPoolExecutor(
                max_workers=worker_count,
                initializer=configure_loggers_on_worker,
                initargs=(log_level, log_queue),
            ) as executor:
                # If we should multiprocess the benchmarking suite
                if worker_count > 1:
                    logging_thread.start()

                    # Record all the futures
                    futures = []

                    # If we should multiprocess the benchmarking suite
                    if accessory_generation_workers_count > 1:
                        # Note: checkpointing will not work correctly when multiprocessing accessory generation
                        # TODO: Change checkpointing to store the indices of the completed accessory generations rather
                        #       than the next index

                        accessories_suppliers = self._accessories_suppliers.copy()

                        generation_futures = []

                        for _ in range(accessory_generation_workers_count):
                            if len(accessories_suppliers) > 0:
                                generation_futures.append(
                                    executor.submit(
                                        BenchmarkSuite._generate_accessories,
                                        gpu_manager,
                                        accessories_suppliers.pop(),
                                    )
                                )

                        i = 0

                        while len(generation_futures) > 0:
                            for future in as_completed(generation_futures):
                                # As soon as one future completes, submit the next one if there are anymore to submit
                                if len(accessories_suppliers) > 0:
                                    generation_futures.append(
                                        executor.submit(
                                            BenchmarkSuite._generate_accessories,
                                            gpu_manager,
                                            accessories_suppliers.pop(),
                                        )
                                    )

                                # Remove the future once we have completed it
                                generation_futures.remove(future)

                                # Increment i
                                i += 1

                                accessories: List[Accessory] = future.result()

                                # For each generated accessory
                                for accessory in accessories:
                                    # Submit benchmarks for each accessory
                                    for factory in self._benchmarks:
                                        LOGGER.info(
                                            "Submitting '%s' benchmark on accessory '%s' with base class '%s' and target '%s' for execution",
                                            factory.name(),
                                            accessory.adversarial_accessory_path,
                                            accessory.base_class,
                                            accessory.target_class,
                                        )
                                        benchmark = factory.construct(
                                            benchmark_arguments, accessory.copy()
                                        )
                                        future = executor.submit(
                                            BenchmarkSuite._process_benchmark,
                                            benchmark,
                                        )
                                        future.add_done_callback(
                                            lambda f: self._future_done_callback(
                                                benchmark_arguments, results, i + 1, f
                                            )
                                        )
                                        futures.append(future)
                    else:
                        # Otherwise, generate the accessories on one process
                        for i, supplier in enumerate(self._accessories_suppliers):
                            accessories = self._generate_accessories(
                                gpu_manager, supplier
                            )
                            # Distribute the accessories across processes
                            for accessory in accessories:
                                # Add the benchmarks for this accessory to the task queue
                                for factory in self._benchmarks:
                                    LOGGER.info(
                                        "Submitting '%s' benchmark on accessory '%s' with base class '%s' and target '%s' for execution",
                                        factory.name(),
                                        accessory.adversarial_accessory_path,
                                        accessory.base_class,
                                        accessory.target_class,
                                    )
                                    benchmark = factory.construct(
                                        benchmark_arguments, accessory.copy()
                                    )
                                    future = executor.submit(
                                        BenchmarkSuite._process_benchmark,
                                        benchmark,
                                    )
                                    future.add_done_callback(
                                        lambda f: self._future_done_callback(
                                            benchmark_arguments, results, i + 1, f
                                        )
                                    )
                                    futures.append(future)

                    # Compile all the processed benchmarks into a list of results
                    for future in as_completed(futures):
                        benchmark_result: BenchmarkResult = future.result()
                        LOGGER.info(
                            "Received benchmark result from '%s' on accessory '%s' to '%s'",
                            benchmark_result.benchmark.name,
                            benchmark_result.benchmark.accessory.base_class,
                            benchmark_result.benchmark.accessory.target_class,
                        )
                        results.append(benchmark_result)
                else:
                    # Execute sequentially
                    for i, supplier in enumerate(self._accessories_suppliers):
                        accessories = supplier.supply(gpu_manager)
                        for accessory in accessories:
                            # Run the benchmark for each accessory
                            for factory in self._benchmarks:
                                benchmark = factory.construct(
                                    benchmark_arguments, accessory.copy()
                                )
                                benchmark_result = self._process_benchmark(
                                    benchmark,
                                )
                                results.append(benchmark_result)

                            self.save_checkpoint(benchmark_arguments, i + 1, results)

            # Stop the logging thread
            log_queue.put(None)

            # Join the logging thread at the end to ensure completion
            if worker_count > 1:
                logging_thread.join()

        LOGGER.info("Benchmark execution completed.")

        # Collate all the statistics from the queue
        self._collate_statistics(results, benchmark_arguments.output_directory)

        end_time = time()

        time_taken = end_time - start_time

        LOGGER.info(
            "Benchmark suite finished in %dm %.02fs",
            int(time_taken // 60),
            time_taken - (time_taken // 60 * 60),
        )

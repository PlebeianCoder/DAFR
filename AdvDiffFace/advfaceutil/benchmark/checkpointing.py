__all__ = [
    "Checkpoint",
    "CheckpointComponent",
    "CheckpointArguments",
]

import pickle
from abc import ABCMeta
from argparse import ArgumentParser
from argparse import Namespace
from dataclasses import dataclass
from logging import getLogger
from pathlib import Path
from time import strftime
from typing import List, Tuple

from advfaceutil.benchmark.args import CheckpointingMode
from advfaceutil.benchmark.suppliers import AccessoriesSupplier
from advfaceutil.benchmark.base import BenchmarkFactory
from advfaceutil.benchmark.suite import BenchmarkResult
from advfaceutil.benchmark.suite import BenchmarkSuite
from advfaceutil.utils import Component
from advfaceutil.utils import ComponentArguments


LATEST_CHECKPOINT_PATH = Path("latest_checkpoint.pickle")
LOGGER = getLogger("checkpointing")


@dataclass(frozen=True)
class CheckpointArguments(ComponentArguments):
    checkpoint_path: Path

    @staticmethod
    def parse_args(args: Namespace) -> "CheckpointArguments":
        checkpoint_path = args.path
        if checkpoint_path is None:
            checkpoint_path = LATEST_CHECKPOINT_PATH

        if not checkpoint_path.exists() and checkpoint_path.is_file():
            raise FileNotFoundError(
                f'Cannot load checkpoint "{checkpoint_path}" as the file does not exist.'
            )

        return CheckpointArguments(checkpoint_path)

    @staticmethod
    def add_args(parser: ArgumentParser) -> None:
        parser.add_argument("--path", help="The path to the checkpoint to load.")


class CheckpointComponent(Component, metaclass=ABCMeta):
    @staticmethod
    def run(args: CheckpointArguments) -> None:
        checkpoint = Checkpoint.load(args.checkpoint_path)

        total_suppliers = len(checkpoint.accessory_suppliers)

        suppliers = checkpoint.accessory_suppliers[checkpoint.next_supplier_index :]

        suite = BenchmarkSuite(checkpoint.benchmarks, suppliers)

        LOGGER.info(
            "Resuming benchmarking suite from checkpoint %s. Starting at supplier %d/%d",
            args.checkpoint_path,
            checkpoint.next_supplier_index,
            total_suppliers,
        )

        suite.run(checkpoint.benchmark_arguments, checkpoint.results)


from advfaceutil.benchmark.args import BenchmarkArguments


CheckpointState = Tuple[
    List[str], BenchmarkArguments, List[AccessoriesSupplier], int, List[BenchmarkResult]
]


@dataclass
class Checkpoint:
    benchmarks: List[BenchmarkFactory]
    benchmark_arguments: BenchmarkArguments
    accessory_suppliers: List[AccessoriesSupplier]
    next_supplier_index: int
    results: List[BenchmarkResult]

    def __getstate__(self) -> CheckpointState:
        return (
            [factory.name() for factory in self.benchmarks],
            self.benchmark_arguments,
            self.accessory_suppliers,
            self.next_supplier_index,
            self.results,
        )

    def __setstate__(self, state: CheckpointState):
        (
            factory_names,
            self.benchmark_arguments,
            self.accessory_suppliers,
            self.next_supplier_index,
            self.results,
        ) = state
        self.benchmarks = [BenchmarkFactory.from_name(name)() for name in factory_names]

    def save(self):
        if self.benchmark_arguments.checkpointing_mode == CheckpointingMode.DISABLED:
            return

        data = pickle.dumps(self)

        # Write the checkpoint to the latest checkpoint path
        LATEST_CHECKPOINT_PATH.write_bytes(data)

        # Also save the checkpoint in the checkpoints directory
        checkpoint_folder = self.benchmark_arguments.output_directory / "checkpoints"
        checkpoint_folder.mkdir(parents=True, exist_ok=True)

        # Write the latest checkpoint file
        (checkpoint_folder / "latest_checkpoint.pickle").write_bytes(data)

        if self.benchmark_arguments.checkpointing_mode == CheckpointingMode.LATEST_ONLY:
            # Write the time stamped file
            file_name = (
                strftime("%Y-%m-%d_%H-%M-%S") + f"-{self.next_supplier_index}.pickle"
            )
            (checkpoint_folder / file_name).write_bytes(data)
        else:
            file_name = "latest_checkpoint.pickle"

        LOGGER.info("Saved checkpoint to %s", checkpoint_folder / file_name)

    @staticmethod
    def load(path: Path) -> "Checkpoint":
        with open(path, "rb") as file:
            return pickle.load(file)

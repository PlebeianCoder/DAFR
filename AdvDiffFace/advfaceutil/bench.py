from argparse import ArgumentParser
from argparse import Namespace
from dataclasses import dataclass

from advfaceutil.benchmark import AccessoryArguments
from advfaceutil.benchmark import BenchmarkArguments
from advfaceutil.benchmark import BenchmarkSuite
from advfaceutil.benchmark import CheckpointArguments
from advfaceutil.benchmark import CheckpointComponent
from advfaceutil.benchmark import from_arguments
from advfaceutil.benchmark import SpecifyBenchmarks
from advfaceutil.utils import Component
from advfaceutil.utils import ComponentArguments
from advfaceutil.utils import ComponentEnum
from advfaceutil.utils import DEFAULT_GLOBAL_ARGUMENTS
from advfaceutil.utils import run


@dataclass(frozen=True)
class RunBenchmarkArguments(ComponentArguments):
    benchmark_arguments: BenchmarkArguments
    accessory_arguments: AccessoryArguments
    specify_benchmarks: SpecifyBenchmarks

    @staticmethod
    def parse_args(args: Namespace) -> "RunBenchmarkArguments":
        return RunBenchmarkArguments(
            BenchmarkArguments.parse_args(args),
            AccessoryArguments.parse_args(args),
            SpecifyBenchmarks.parse_args(args),
        )

    @staticmethod
    def add_args(parser: ArgumentParser) -> None:
        BenchmarkArguments.add_args(parser)
        AccessoryArguments.add_args(parser)
        SpecifyBenchmarks.add_args(parser)


class RunBenchmarkComponent(Component):
    @staticmethod
    def run(args: RunBenchmarkArguments) -> None:
        benchmark = BenchmarkSuite(
            args.specify_benchmarks.benchmarks,
            from_arguments(args.accessory_arguments, args.benchmark_arguments),
        )

        benchmark.run(args.benchmark_arguments)


class BenchmarkComponents(ComponentEnum):
    RUN = (RunBenchmarkArguments, RunBenchmarkComponent)
    LOAD = (CheckpointArguments, CheckpointComponent)


if __name__ == "__main__":
    run(BenchmarkComponents, DEFAULT_GLOBAL_ARGUMENTS)

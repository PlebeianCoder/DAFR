__all__ = [
    "BenchmarkArgument",
    "BenchmarkArguments",
    "AccessoryArguments",
    "SpecifyBenchmarks",
    "SpecifyPairwiseClassNames",
    "CheckpointingMode",
]

from argparse import ArgumentParser
from argparse import Namespace
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from time import strftime
from typing import Optional, Any, List, Dict, Type

from advfaceutil.datasets import FaceDatasetSize
from advfaceutil.datasets import FaceDatasets
from advfaceutil.recognition import RecognitionArchitectures
from advfaceutil.recognition.processing import FaceProcessors
from advfaceutil.utils import ComponentArguments


class CheckpointingMode(Enum):
    DISABLED = "disabled"
    LATEST_ONLY = "latest_only"
    ALL = "all"


class BenchmarkArgument(Enum):
    DATASET_DIRECTORY = auto()
    RESEARCHERS_DIRECTORY = auto()
    OUTPUT_DIRECTORY = auto()
    ARCHITECTURE = auto()
    DATASET = auto()
    SIZE = auto()
    FACE_PROCESSOR = auto()
    ADDITIVE_OVERLAY = auto()
    OVERLAY_MODEL = auto()
    WEIGHTS_DIRECTORY = auto()
    SAVE_AUGMENTED_IMAGES = auto()
    SAVE_ALIGNED_IMAGES = auto()
    CLASS_IMAGE_LIMIT = auto()
    SAVE_RAW_STATISTICS = auto()
    WORKER_COUNT = auto()
    ACCESSORY_GENERATION_WORKERS_COUNT = auto()
    ACCESSORY_GENERATION_PROCESSES_PER_GPU = auto()
    CHECKPOINTING_MODE = auto()


@dataclass(frozen=True)
class BenchmarkArguments(ComponentArguments):
    # The dataset containing the PubFig images
    dataset_directory: Path
    # The dataset containing the researchers images
    researchers_directory: Path
    # The output directory
    output_directory: Path
    # The recognition architecture to use
    architecture: RecognitionArchitectures
    # The dataset to use
    dataset: FaceDatasets
    # The size of the dataset to use
    size: FaceDatasetSize
    # The face processor to use to augment the images
    face_processor: FaceProcessors

    # This is optional and is only used if the face processor is dlib
    additive_overlay: bool = False
    # This is optional and is only used if the face processor is mediapipe
    overlay_model: Optional[Path] = None

    # The weights directory for the recognition model
    weights_directory: Optional[Path] = None

    # Optionally save the augmented images
    save_augmented_images: bool = False
    # Optionally save aligned images
    save_aligned_images: bool = False

    # Optionally limit the number of images per class
    class_image_limit: Optional[int] = 100

    # Whether to save the raw statistics
    save_raw_statistics: bool = False

    # The maximum number of workers to use when executing a benchmark suite
    worker_count: Optional[int] = None

    # The number of workers to divide accessory generation over
    accessory_generation_workers_count: int = 1

    # The number of accessory generation processes that can use the same GPU
    accessory_generation_processes_per_gpu: int = 1

    # Control the saving method we use for saving checkpoints
    checkpointing_mode: CheckpointingMode = CheckpointingMode.LATEST_ONLY

    @staticmethod
    def from_defaults(
        default_values: Dict[BenchmarkArgument, Any]
    ) -> Type["BenchmarkArguments"]:
        # This will create benchmark arguments with default values and will not add the default values to the parser
        class BenchmarkArgumentsFromDefaults(BenchmarkArguments):
            @staticmethod
            def parse_args(args: Namespace) -> "BenchmarkArguments":
                if BenchmarkArgument.DATASET_DIRECTORY not in default_values:
                    dataset_directory = Path(args.dataset_directory)
                    if not dataset_directory.exists() or not dataset_directory.is_dir():
                        raise Exception(
                            f"Dataset directory must be a valid directory but was given {dataset_directory}"
                        )
                else:
                    dataset_directory = default_values[
                        BenchmarkArgument.DATASET_DIRECTORY
                    ]
                    assert isinstance(dataset_directory, Path)
                    assert dataset_directory.exists() and dataset_directory.is_dir()

                if BenchmarkArgument.RESEARCHERS_DIRECTORY not in default_values:
                    researchers_directory = Path(args.researchers_directory)
                    if (
                        not researchers_directory.exists()
                        or not researchers_directory.is_dir()
                    ):
                        raise Exception(
                            f"Researchers directory must be a valid directory but was given {researchers_directory}"
                        )
                else:
                    researchers_directory = default_values[
                        BenchmarkArgument.RESEARCHERS_DIRECTORY
                    ]
                    assert isinstance(researchers_directory, Path)
                    assert (
                        researchers_directory.exists()
                        and researchers_directory.is_dir()
                    )

                if BenchmarkArgument.OUTPUT_DIRECTORY not in default_values:
                    output_directory = Path(args.output_directory)
                    # Set the output directory to be a sub-folder of the given directory for the
                    # current date and time
                    output_directory = output_directory / strftime("%Y-%m-%d_%H-%M-%S")
                    # Create the directory if it doesn't exist
                    output_directory.mkdir(parents=True, exist_ok=True)
                else:
                    output_directory = default_values[
                        BenchmarkArgument.OUTPUT_DIRECTORY
                    ]
                    assert isinstance(output_directory, Path)
                    output_directory.mkdir(parents=True, exist_ok=True)

                if BenchmarkArgument.ARCHITECTURE not in default_values:
                    architecture = RecognitionArchitectures[args.architecture]
                else:
                    architecture = default_values[BenchmarkArgument.ARCHITECTURE]
                    assert isinstance(architecture, RecognitionArchitectures)

                if BenchmarkArgument.DATASET not in default_values:
                    dataset = FaceDatasets[args.dataset]
                else:
                    dataset = default_values[BenchmarkArgument.DATASET]
                    assert isinstance(dataset, FaceDatasets)

                if BenchmarkArgument.SIZE not in default_values:
                    size = dataset.get_size(args.size)
                else:
                    size = default_values[BenchmarkArgument.SIZE]
                    assert isinstance(size, FaceDatasetSize)

                if BenchmarkArgument.FACE_PROCESSOR not in default_values:
                    face_processor = FaceProcessors[args.face_processor]
                else:
                    face_processor = default_values[BenchmarkArgument.FACE_PROCESSOR]
                    assert isinstance(face_processor, FaceProcessors)

                if BenchmarkArgument.ADDITIVE_OVERLAY not in default_values:
                    additive_overlay = bool(args.additive_overlay)
                else:
                    additive_overlay = default_values[
                        BenchmarkArgument.ADDITIVE_OVERLAY
                    ]
                    assert isinstance(additive_overlay, bool)

                if BenchmarkArgument.OVERLAY_MODEL not in default_values:
                    overlay_model = (
                        Path(args.overlay_model) if args.overlay_model else None
                    )
                    if overlay_model is not None and (
                        not overlay_model.exists() or not overlay_model.is_file()
                    ):
                        raise Exception(
                            f"Overlay model path must exist but was given {overlay_model}"
                        )
                else:
                    overlay_model = default_values[BenchmarkArgument.OVERLAY_MODEL]
                    assert overlay_model is None or (
                        isinstance(overlay_model, Path)
                        and overlay_model.exists()
                        and overlay_model.is_file()
                    )

                if BenchmarkArgument.WEIGHTS_DIRECTORY not in default_values:
                    if args.weights_directory:
                        weights_directory = Path(args.weights_directory)
                        if (
                            not weights_directory.exists()
                            or not weights_directory.is_dir()
                        ):
                            raise Exception(
                                f"Pretrained weights directory must exist but was given {weights_directory}"
                            )
                    else:
                        weights_directory = None
                else:
                    weights_directory = default_values[
                        BenchmarkArgument.WEIGHTS_DIRECTORY
                    ]
                    assert weights_directory is None or (
                        isinstance(weights_directory, Path)
                        and weights_directory.exists()
                        and weights_directory.is_dir()
                    )

                if BenchmarkArgument.SAVE_AUGMENTED_IMAGES not in default_values:
                    save_augmented_images = bool(args.save_augmented_images)
                else:
                    save_augmented_images = default_values[
                        BenchmarkArgument.SAVE_AUGMENTED_IMAGES
                    ]
                    assert isinstance(save_augmented_images, bool)

                if BenchmarkArgument.SAVE_ALIGNED_IMAGES not in default_values:
                    save_aligned_images = bool(args.save_aligned_images)
                else:
                    save_aligned_images = default_values[
                        BenchmarkArgument.SAVE_ALIGNED_IMAGES
                    ]
                    assert isinstance(save_aligned_images, bool)

                if BenchmarkArgument.CLASS_IMAGE_LIMIT not in default_values:
                    if bool(args.no_class_image_limit):
                        class_image_limit = None
                    else:
                        class_image_limit = args.class_image_limit
                else:
                    class_image_limit = default_values[
                        BenchmarkArgument.CLASS_IMAGE_LIMIT
                    ]
                    assert class_image_limit

                if BenchmarkArgument.SAVE_RAW_STATISTICS not in default_values:
                    save_raw_statistics = bool(args.save_raw_statistics)
                else:
                    save_raw_statistics = default_values[
                        BenchmarkArgument.SAVE_RAW_STATISTICS
                    ]
                    assert isinstance(save_raw_statistics, bool)

                if BenchmarkArgument.WORKER_COUNT not in default_values:
                    worker_count = args.workers
                else:
                    worker_count = default_values[BenchmarkArgument.WORKER_COUNT]
                    assert isinstance(worker_count, int)

                if (
                    BenchmarkArgument.ACCESSORY_GENERATION_WORKERS_COUNT
                    not in default_values
                ):
                    accessory_generation_workers_count = (
                        args.accessory_generation_workers_count
                    )
                else:
                    accessory_generation_workers_count = default_values[
                        BenchmarkArgument.ACCESSORY_GENERATION_WORKERS_COUNT
                    ]
                    assert isinstance(accessory_generation_workers_count, int)

                if (
                    BenchmarkArgument.ACCESSORY_GENERATION_PROCESSES_PER_GPU
                    not in default_values
                ):
                    accessory_generation_processes_per_gpu = (
                        args.accessory_generation_processes_per_gpu
                    )
                else:
                    accessory_generation_processes_per_gpu = default_values[
                        BenchmarkArgument.ACCESSORY_GENERATION_PROCESSES_PER_GPU
                    ]
                    assert isinstance(accessory_generation_processes_per_gpu, int)

                if BenchmarkArgument.CHECKPOINTING_MODE not in default_values:
                    checkpointing_mode = CheckpointingMode[args.checkpointing_mode]
                else:
                    checkpointing_mode = default_values[
                        BenchmarkArgument.CHECKPOINTING_MODE
                    ]
                    assert isinstance(checkpointing_mode, CheckpointingMode)

                return BenchmarkArguments(
                    dataset_directory,
                    researchers_directory,
                    output_directory,
                    architecture,
                    dataset,
                    size,
                    face_processor,
                    additive_overlay,
                    overlay_model,
                    weights_directory,
                    save_augmented_images,
                    save_aligned_images,
                    class_image_limit,
                    save_raw_statistics,
                    worker_count,
                    accessory_generation_workers_count,
                    accessory_generation_processes_per_gpu,
                    checkpointing_mode,
                )

            @staticmethod
            def add_args(parser: ArgumentParser) -> None:
                if BenchmarkArgument.DATASET_DIRECTORY not in default_values:
                    parser.add_argument(
                        "dataset_directory",
                        type=str,
                        help="The directory for the dataset images",
                    )

                if BenchmarkArgument.RESEARCHERS_DIRECTORY not in default_values:
                    parser.add_argument(
                        "researchers_directory",
                        type=str,
                        help="The directory for the researchers images",
                    )

                if BenchmarkArgument.OUTPUT_DIRECTORY not in default_values:
                    parser.add_argument(
                        "output_directory", type=str, help="The output directory"
                    )

                if BenchmarkArgument.ARCHITECTURE not in default_values:
                    parser.add_argument(
                        "architecture",
                        type=str,
                        choices=[arch.name for arch in RecognitionArchitectures],
                        help="The recognition architecture to use",
                    )

                if BenchmarkArgument.DATASET not in default_values:
                    parser.add_argument(
                        "dataset",
                        type=str,
                        choices=[d.name for d in FaceDatasets],
                        help="The dataset to use",
                    )

                if BenchmarkArgument.SIZE not in default_values:
                    parser.add_argument(
                        "size",
                        type=str,
                        choices=["SMALL", "LARGE"],
                        help="The size of the dataset to use",
                    )

                if BenchmarkArgument.FACE_PROCESSOR not in default_values:
                    parser.add_argument(
                        "face_processor",
                        type=str,
                        choices=[processor.name for processor in FaceProcessors],
                        help="The face processor to use to augment the images",
                    )

                if BenchmarkArgument.ADDITIVE_OVERLAY not in default_values:
                    parser.add_argument(
                        "-ao",
                        "--additive-overlay",
                        action="store_true",
                        help="This is optional and is only used if the face processor is dlib",
                    )

                if BenchmarkArgument.OVERLAY_MODEL not in default_values:
                    parser.add_argument(
                        "-om",
                        "--overlay-model",
                        type=str,
                        help="This is optional and is only used if the face processor is mediapipe",
                    )

                if BenchmarkArgument.WEIGHTS_DIRECTORY not in default_values:
                    parser.add_argument(
                        "-wd",
                        "--weights-directory",
                        type=str,
                        help="The weights directory for the recognition model",
                    )

                if BenchmarkArgument.SAVE_AUGMENTED_IMAGES not in default_values:
                    parser.add_argument(
                        "-sau",
                        "--save-augmented-images",
                        action="store_true",
                        help="Optionally save the augmented images",
                    )

                if BenchmarkArgument.SAVE_ALIGNED_IMAGES not in default_values:
                    parser.add_argument(
                        "-sal",
                        "--save-aligned-images",
                        action="store_true",
                        help="Optionally save aligned images",
                    )

                if BenchmarkArgument.CLASS_IMAGE_LIMIT not in default_values:
                    parser.add_argument(
                        "--no-class-image-limit",
                        action="store_true",
                        help="Choose not to limit the number of images per class",
                    )
                    parser.add_argument(
                        "--class-image-limit",
                        type=int,
                        default=100,
                        help="Optionally limit the number of images per class",
                    )

                if BenchmarkArgument.SAVE_RAW_STATISTICS not in default_values:
                    parser.add_argument(
                        "-srs",
                        "--save-raw-statistics",
                        action="store_true",
                        help="Optionally save the raw statistics",
                    )

                if BenchmarkArgument.WORKER_COUNT not in default_values:
                    parser.add_argument(
                        "-wc",
                        "--workers",
                        type=int,
                        help="Optionally set the number of workers to use to execute the benchmark. Using 0 will cause"
                        "the benchmark to execute sequentially. Not setting this will use as many processes as "
                        "possible",
                    )

                if (
                    BenchmarkArgument.ACCESSORY_GENERATION_WORKERS_COUNT
                    not in default_values
                ):
                    parser.add_argument(
                        "-agwc",
                        "--accessory-generation-workers-count",
                        type=int,
                        default=1,
                        help="The number of workers to divide accessory generation over. By default this is 1 (i.e. use"
                        "only one process to generate accessories. This is determined experimentally and is"
                        "usually limited by the VRAM of the GPUs in use. This value is dependent on the number of"
                        "processes per GPU and the number of GPUs available.",
                    )

                if (
                    BenchmarkArgument.ACCESSORY_GENERATION_PROCESSES_PER_GPU
                    not in default_values
                ):
                    parser.add_argument(
                        "-agppg",
                        "--accessory-generation-processes-per-gpu",
                        type=int,
                        default=1,
                        help="The number of processes that can use the same GPU during accessory generation.",
                    )

                if BenchmarkArgument.CHECKPOINTING_MODE not in default_values:
                    parser.add_argument(
                        "-cpm",
                        "--checkpointing-mode",
                        type=str,
                        choices=[mode.name for mode in CheckpointingMode],
                        default=CheckpointingMode.LATEST_ONLY.name,
                        help="Control the saving method we use for saving checkpoints. By default we only save the"
                        "latest checkpoint but this can be configured to save all or no checkpoints.",
                    )

        return BenchmarkArgumentsFromDefaults

    @staticmethod
    def parse_args(args: Namespace) -> "BenchmarkArguments":
        return BenchmarkArguments.from_defaults({}).parse_args(args)

    @staticmethod
    def add_args(parser: ArgumentParser) -> None:
        BenchmarkArguments.from_defaults({}).add_args(parser)


@dataclass
class AccessoryArguments(ComponentArguments):
    # The path to the accessory to add to the images
    adversarial_accessory_path: Path
    # The path to the non-adversarial accessory
    non_adversarial_accessory_path: Path
    # The class that the accessory was designed for
    base_class: str
    # The class that we are supposed to target
    target_class: Optional[str] = None
    # The index of the base class
    base_class_index: Optional[int] = None
    # The index of the target class
    target_class_index: Optional[int] = None

    # Optionally define the classes that we should use for universal impersonation
    universal_class_indices: Optional[List[int]] = None

    @staticmethod
    def parse_args(args: Namespace) -> "AccessoryArguments":
        accessory_path = Path(args.accessory_path)
        if not accessory_path.exists() or not accessory_path.is_file():
            raise Exception(f"Accessory path must exist but was given {accessory_path}")
        non_adversarial_accessory_path = Path(args.non_adversarial_accessory_path)
        if (
            not non_adversarial_accessory_path.exists()
            or not non_adversarial_accessory_path.is_file()
        ):
            raise Exception(
                f"Non-adversarial accessory path must exist but was given {non_adversarial_accessory_path}"
            )
        base_class = args.base_class
        target_class = args.target_class

        return AccessoryArguments(
            accessory_path,
            non_adversarial_accessory_path,
            base_class,
            target_class,
        )

    @staticmethod
    def add_args(parser: ArgumentParser) -> None:
        parser.add_argument(
            "accessory_path",
            type=str,
            help="The path to the accessory to add to the images",
        )
        parser.add_argument(
            "non_adversarial_accessory_path",
            type=str,
            help="The path to the non-adversarial accessory",
        )
        parser.add_argument(
            "base_class",
            type=str,
            help="The class that the accessory was designed for",
        )
        parser.add_argument(
            "-to",
            "--target-class",
            type=str,
            help="The class that we are supposed to target",
        )


from advfaceutil.benchmark.base import BenchmarkFactory


@dataclass
class SpecifyBenchmarks(ComponentArguments):
    benchmarks: List[BenchmarkFactory]

    def __getstate__(self) -> Any:
        return [factory.name() for factory in self.benchmarks]

    def __setstate__(self, state: List[str]) -> None:
        self.benchmarks = [BenchmarkFactory.from_name(name)() for name in state]

    @staticmethod
    def parse_args(args: Namespace) -> "SpecifyBenchmarks":
        benchmark_strings = args.benchmark

        if benchmark_strings is None or len(benchmark_strings) == 0:
            from advfaceutil.benchmark.benchmarks import DEFAULT_BENCHMARKS

            return SpecifyBenchmarks(DEFAULT_BENCHMARKS)

        benchmarks = []
        for benchmark_name in benchmark_strings:
            benchmarks.append(BenchmarkFactory.from_name(benchmark_name)())

        return SpecifyBenchmarks(benchmarks)

    @staticmethod
    def add_args(parser: ArgumentParser) -> None:
        parser.add_argument(
            "-b",
            "--benchmark",
            type=str,
            action="append",
            help="Specify a benchmark to use. For example: 'Baseline' will use the baseline benchmark.",
        )


@dataclass(frozen=True)
class SpecifyPairwiseClassNames(ComponentArguments):
    class_names: List[str]

    @staticmethod
    def parse_args(args: Namespace) -> "SpecifyPairwiseClassNames":
        universal_class_names = args.add_class
        return SpecifyPairwiseClassNames(universal_class_names)

    @staticmethod
    def add_args(parser: ArgumentParser) -> None:
        parser.add_argument(
            "-ac",
            "--add-class",
            type=str,
            nargs="+",
            help="Add a class name that will be used for universal impersonation. This can be used multiple times to "
            "add multiple class names.",
        )

from argparse import ArgumentParser
from argparse import Namespace
from dataclasses import dataclass
from logging import getLogger
from pathlib import Path
from typing import Optional
from functools import cached_property

import os
import re
import cv2

# DEBUG
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import torch

from advfaceutil.benchmark import GPUDeviceManager
from advfaceutil.benchmark import Accessory
from advfaceutil.benchmark import AccessoryGenerator
from advfaceutil.benchmark import BenchmarkArgument
from advfaceutil.benchmark import BenchmarkArguments
from advfaceutil.benchmark import BenchmarkSuite
from advfaceutil.benchmark import pairwise_impersonation, dodge_each, random_dodge_each
from advfaceutil.benchmark import random_pairwise_impersonation
from advfaceutil.benchmark import SpecifyBenchmarks
from advfaceutil.datasets import FaceDatasets
from advfaceutil.datasets import FaceDatasetSize
from advfaceutil.datasets import IndividualDataset
from advfaceutil.utils import Component
from advfaceutil.utils import ComponentArguments
from advfaceutil.utils import run_component
from advfaceutil.utils import DEFAULT_GLOBAL_ARGUMENTS
from config import patch_config_types
import numpy as np
from torchvision.utils import save_image
from advfaceutil.recognition.processing import FaceProcessors, DlibFaceProcessor, MediaPipeFaceProcessor

from train import AdversarialMask

DefaultedBenchmarkArguments = BenchmarkArguments.from_defaults(
    {
        BenchmarkArgument.ADDITIVE_OVERLAY: False,
        BenchmarkArgument.FACE_PROCESSOR: FaceProcessors.ADV,
        BenchmarkArgument.OVERLAY_MODEL: None
    }
)

def str2Bool(s):
    return s.lower()=="true"

@dataclass(frozen=True)
class AdvMaskBenchmarkArguments(ComponentArguments):
    initial_mask: str
    tv_weight:float
    dodge:bool
    # Dataset directory here is different than the ones in the general benchmark arguments
    # These ones are for generation, ones in general benchmark are for testing
    gen_dataset_directory: Path # Have to be unaligned
    gen_researchers_directory: Path # Have to be unaligned
    benchmark_arguments: BenchmarkArguments
    specify_benchmark: SpecifyBenchmarks

    @staticmethod
    def parse_args(args:Namespace) -> "AdvMaskBenchmarkArguments":
        initial_mask = args.initial_mask
        tv_weight = args.tv_weight
        dodge = args.dodge
        gen_dataset_directory = Path(args.gen_dataset_directory)
        if (
            not gen_dataset_directory.exists()
            or not gen_dataset_directory.is_dir()
        ):
            raise Exception(
                f"Dataset directory must be a valid directory but was given {gen_dataset_directory}"
            )
        gen_researchers_directory = Path(args.gen_researchers_directory)
        if (
            not gen_researchers_directory.exists()
            or not gen_researchers_directory.is_dir()
        ):
            raise Exception(
                f"Researchers directory must be a valid directory but was given {gen_researchers_directory}"
            )
    
        benchmark_arguments = DefaultedBenchmarkArguments.parse_args(args)
        specify_benchmark = SpecifyBenchmarks.parse_args(args)

        return  AdvMaskBenchmarkArguments(
            initial_mask,
            tv_weight,
            dodge,
            gen_dataset_directory,
            gen_researchers_directory,
            benchmark_arguments,
            specify_benchmark
        )

    @staticmethod
    def add_args(parser: ArgumentParser) -> None:
        parser.add_argument(
            "-init",
            "--initial-mask",
            type=str,
            help="Determines the initial mask of the attack."
        )
        parser.add_argument(
            "-tv",
            "--tv-weight",
            type=float,
            help="The TV weight used in the attack."
        )
        parser.add_argument(
            "-ddg",
            "--dodge",
            type=str2Bool,
            help="Whether to perform dodging or impersonation"
        )
        parser.add_argument(
            "-gddir",
            "--gen-dataset-directory",
            type=str,
            help="The path that points to the directory containing the dataset pictures. These images will be used to generate the adversarial mask."
        )
        parser.add_argument(
            "-grdir",
            "--gen-researchers-directory",
            type=str,
            help="The path that points to the directory containing the researchers pictures. These images will be used to generate the adversarial mask."
        )
        DefaultedBenchmarkArguments.add_args(parser)
        SpecifyBenchmarks.add_args(parser)


class AdvMaskAccessoryGenerator(AccessoryGenerator):
    def __init__(self, args: AdvMaskBenchmarkArguments) -> None:
        self.args = args
        self.output_directory = args.benchmark_arguments.output_directory
        self.output_directory.mkdir(parents=True, exist_ok=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.architecture = self.args.benchmark_arguments.architecture.construct(
            self.args.benchmark_arguments.dataset,
            self.args.benchmark_arguments.size,
            self.args.benchmark_arguments.weights_directory,
            self.device)

        self.initial_mask = args.initial_mask
        self.tv_weight = args.tv_weight
        rnetType = "r18"
        if "r34" in str(self.args.benchmark_arguments.weights_directory):
            rnetType = "r34"
        elif "r50" in str(self.args.benchmark_arguments.weights_directory):
            rnetType = "r50"
        elif "r100" in str(self.args.benchmark_arguments.weights_directory):
            rnetType = "r100"
        elif "fted100" in str(self.args.benchmark_arguments.weights_directory):
            rnetType = "fted100"
        elif "clip" in str(self.args.benchmark_arguments.weights_directory):
            rnetType = "farl"
        
        
        datasetName = "PUBFIG"
        if self.args.benchmark_arguments.dataset == FaceDatasets.VGGFACE2:
            datasetName = "VGGFACE2"
        self.anchorBase = f"../anchors/masked_{rnetType}_{datasetName}_"

    def generate(
        self,
        gpu_manager: GPUDeviceManager,
        dataset,
        size: FaceDatasetSize,
        base_class: int,
        target_class: Optional[int],
        base_class_name:str,
        target_class_name: Optional[str],
        universal_class_indices):

        logger = getLogger("AdvMask_benchmark")
        # names = IndividualDataset(...)

        logger.info("Getting GPU")

        # device = gpu_manager.acquire()
        logger.info("Loading network")
        logger.info("Architecture loaded")

        celebDir = self.args.gen_dataset_directory.as_posix()
        resDir = self.args.gen_researchers_directory.as_posix()
        baseClassName = size.class_names[base_class]

        anchorPic = ""
        if target_class_name is None:
            anchorPic = f"{self.anchorBase}{base_class_name}.pth"
        else:
            anchorPic = f"{self.anchorBase}{target_class_name}.pth"

        logger.info(f"Using anchor of {anchorPic}")


        save_name = target_class_name 
        if target_class is None:
            save_name = "None"

        config = patch_config_types["targeted"](celebDir, [base_class_name])

        curMasks = AdversarialMask(config, self.architecture, anchorPic, 
            os.path.join(self.output_directory, f"NonAdv_{base_class_name}To{save_name}.png"), 
            False, self.initial_mask, self.tv_weight, target_class_name is not None)


        logger.info("Non Adv Being Created")
        # Non adversarial generation first, negative steps so that no adv is ever done
        initial_m = curMasks.train(True)
        non_adv_accessory_path = Path(os.path.join(self.output_directory, f"NonAdv_{base_class_name}To{save_name}.png"))

        logger.info("Adv Being Created")
        
        adv_accessory_path = Path(os.path.join(self.output_directory, f"AdvAdv_{base_class_name}To{save_name}.png"))
        curMasks.change_outpath(adv_accessory_path.as_posix())
        curMasks.train(False, initial_m)
        logger.info("Adv Created GPU Being Released")
        # Release the GPU after generation
        # gpu_manager.release()

        return [Accessory(
            adv_accessory_path,
            non_adv_accessory_path,
            base_class_name,
            target_class_name,
            base_class,
            target_class,
            universal_class_indices
        )]

class AdvMaskBenchmark(Component):
    @classmethod
    def run(cls, args: AdvMaskBenchmarkArguments) -> None:
        if args.dodge:
            class_names = ['GeorgeClooney', 'ReeseWitherspoon', 'HughGrant', 'DrewBarrymore', 'AngelinaJolie', 'ColinPowell', 'OrlandoBloom', 'JenniferLopez', 'KeiraKnightley', 'JenniferAniston', 'ReneeZellweger', 'GwynethPaltrow', 'JessicaSimpson', 'LeonardoDiCaprio', 'AliciaKeys', 'JodieFoster', 'TomCruise', 'MattDamon', 'NicoleKidman', 'CateBlanchett', 'SalmaHayek', 'EvaMendes', 'WillSmith', 'AvrilLavigne', 'CameronDiaz', 'JohnTravolta', 'DavidBeckham', 'CharlizeTheron', 'HalleBerry', 'BeyonceKnowles']
            if args.benchmark_arguments.dataset == FaceDatasets.VGGFACE2:
                class_names = ["n000225", "n001145", "n002141", "n003459", "n003485", "n004457", "n004508", "n006214", "n007229", "n007564"]
            accessory_generator = random_dodge_each(
                args.benchmark_arguments.dataset,
                args.benchmark_arguments.size,
                AdvMaskAccessoryGenerator(args),
                class_count=30,
                class_names=class_names
            )

        else:
            chosenBase =['n005219', 'n000266', 'n004216', 'n004538', 'n000176']
            chosenTargets = ['n001765', 'n004589', 'n004737', 'n002886', 'n007139', 'n001586']
            accessory_generator = random_pairwise_impersonation(
                args.benchmark_arguments.dataset,
                args.benchmark_arguments.size,
                AdvMaskAccessoryGenerator(args),
                class_count=6,
                class_names=chosenTargets,
                researchers_only=False,
                base_class_names = chosenBase
            )

        suite = BenchmarkSuite(args.specify_benchmark.benchmarks, accessory_generator)

        suite.run(args.benchmark_arguments)



if __name__ == "__main__":
    run_component(AdvMaskBenchmark, AdvMaskBenchmarkArguments, DEFAULT_GLOBAL_ARGUMENTS)
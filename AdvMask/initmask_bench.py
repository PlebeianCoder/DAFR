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
class InitMaskBenchmarkArguments(ComponentArguments):
    initial_mask: str
    dodge:bool
    benchmark_arguments: BenchmarkArguments
    specify_benchmark: SpecifyBenchmarks

    @staticmethod
    def parse_args(args:Namespace) -> "InitMaskBenchmarkArguments":
        initial_mask = args.initial_mask
        dodge = args.dodge
        benchmark_arguments = DefaultedBenchmarkArguments.parse_args(args)
        specify_benchmark = SpecifyBenchmarks.parse_args(args)

        return  InitMaskBenchmarkArguments(
            initial_mask,
            dodge,
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
            "-ddg",
            "--dodge",
            type=str2Bool,
            help="Whether to perform dodging or impersonation"
        )
        DefaultedBenchmarkArguments.add_args(parser)
        SpecifyBenchmarks.add_args(parser)


class InitMaskAccessoryGenerator(AccessoryGenerator):
    def __init__(self, args: InitMaskBenchmarkArguments) -> None:
        self.args = args
        self.output_directory = args.benchmark_arguments.output_directory
        self.output_directory.mkdir(parents=True, exist_ok=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.initial_mask = args.initial_mask
        if self.initial_mask == 'random':
            self.patch = torch.rand((1, 3, 112, 112), dtype=torch.float32)
        elif self.initial_mask == 'white':
            self.patch = torch.ones((1, 3, 112, 112), dtype=torch.float32)
        elif self.initial_mask == 'black':
            self.patch = torch.zeros((1, 3, 112, 112), dtype=torch.float32) + 0.01
        else:
            print("No valid intiial mask was given")

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

        non_adv = os.path.join(self.output_directory, f"NonAdv.png")
        adv = os.path.join(self.output_directory, f"Adv_{base_class_name}To{target_class_name}.png")
        if not os.path.exists(non_adv):
            save_image(self.patch , non_adv)
        if not os.path.exists(adv):
            save_image(self.patch , adv)
        
        return [Accessory(
            Path(adv),
            Path(non_adv),
            base_class_name,
            target_class_name,
            base_class,
            target_class,
            universal_class_indices
        )]

class InitMaskBenchmark(Component):
    @classmethod
    def run(cls, args: InitMaskBenchmarkArguments) -> None:
        if args.dodge:
            class_names = ['GeorgeClooney', 'ReeseWitherspoon', 'HughGrant', 'DrewBarrymore', 'AngelinaJolie', 'ColinPowell', 'OrlandoBloom', 'JenniferLopez', 'KeiraKnightley', 'JenniferAniston', 'ReneeZellweger', 'GwynethPaltrow', 'JessicaSimpson', 'LeonardoDiCaprio', 'AliciaKeys', 'JodieFoster', 'TomCruise', 'MattDamon', 'NicoleKidman', 'CateBlanchett', 'SalmaHayek', 'EvaMendes', 'WillSmith', 'AvrilLavigne', 'CameronDiaz', 'JohnTravolta', 'DavidBeckham', 'CharlizeTheron', 'HalleBerry', 'BeyonceKnowles']
            if args.benchmark_arguments.dataset == FaceDatasets.VGGFACE2:
                class_names = ["n000225", "n001145", "n002141", "n003459", "n003485", "n004457", "n004508", "n006214", "n007229", "n007564"]
            accessory_generator = random_dodge_each(
                args.benchmark_arguments.dataset,
                args.benchmark_arguments.size,
                InitMaskAccessoryGenerator(args),
                class_count=30,
                class_names=class_names
            )

        else:
            chosenBase =['n005219', 'n000266', 'n004216', 'n004538', 'n000176']
            chosenTargets = ['n001765', 'n004589', 'n004737', 'n002886', 'n007139', 'n001586']
            accessory_generator = random_pairwise_impersonation(
                args.benchmark_arguments.dataset,
                args.benchmark_arguments.size,
                InitMaskAccessoryGenerator(args),
                class_count=6,
                class_names=chosenTargets,
                researchers_only=False,
                base_class_names = chosenBase
            )

        suite = BenchmarkSuite(args.specify_benchmark.benchmarks, accessory_generator)

        suite.run(args.benchmark_arguments)



if __name__ == "__main__":
    run_component(InitMaskBenchmark, InitMaskBenchmarkArguments, DEFAULT_GLOBAL_ARGUMENTS)
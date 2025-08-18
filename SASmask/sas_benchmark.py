from argparse import ArgumentParser
from argparse import Namespace
from dataclasses import dataclass
from logging import getLogger
from pathlib import Path
from typing import Optional

import glob
import os
import re

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
from advfaceutil.datasets import FaceDataset, FaceDatasets
from advfaceutil.datasets import FaceDatasetSize
from advfaceutil.datasets import IndividualDataset
from advfaceutil.utils import Component
from advfaceutil.utils import ComponentArguments
from advfaceutil.utils import run_component
from advfaceutil.utils import DEFAULT_GLOBAL_ARGUMENTS

import numpy as np

from advfaceutil.recognition.processing import FaceProcessors
from train_stymask import train_stymask

DefaultedBenchmarkArguments = BenchmarkArguments.from_defaults(
    {
        BenchmarkArgument.ADDITIVE_OVERLAY: False,
        BenchmarkArgument.FACE_PROCESSOR: FaceProcessors.ADV,
        BenchmarkArgument.OVERLAY_MODEL: Path("../DAFR-v2/3Dobjs/Face_Mask/face_mask.obj")
    }
    # Just to stop errors, the overlay will not be used
)

def str2Bool(s):
    return s.lower()=="true"
    

@dataclass(frozen=True)
class SASBenchmarkArguments(ComponentArguments):
    
    dist_weight:float
    l1_weight: float
    percept_weight: float
    tv_weight: float
    style_weight:float
    withStyle:bool
    threshold:float
    epoch: int
    model_lr:float
    dodge:bool
    testStyle:bool
    uv_path: Path
    style_path: Path
    # Dataset directory here is different than the ones in the general benchmark arguments
    # These ones are for generation, ones in general benchmark are for testing
    gen_dataset_directory: Path # Have to be unaligned
    gen_researchers_directory: Path # Have to be unaligned
    benchmark_arguments: BenchmarkArguments
    specify_benchmark: SpecifyBenchmarks

    @staticmethod
    def parse_args(args:Namespace) -> "SASBenchmarkArguments":
        # print(dir(args))
        dist_weight = args.dist_weight
        l1_weight = args.l1_weight
        percept_weight = args.percept_weight
        tv_weight = args.tv_weight
        style_weight = args.style_weight
        withStyle=args.withStyle
        threshold=args.threshold
        epoch = args.epoch
        model_lr = args.model_lr
        dodge = args.dodge
        testStyle = args.testStyle
        uv_path = Path(args.uv_path)
        if (
            not uv_path.exists()
        ):
            raise Exception(
                f"Path to the UV mask must exist"
            )
        style_path = Path(args.style_path)
        if (
            not style_path.exists()
        ):
            raise Exception(
                f"Path to the style image must exist"
            )
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

        return  SASBenchmarkArguments(
            dist_weight,
            l1_weight, 
            percept_weight,
            tv_weight,
            style_weight,
            withStyle,
            threshold,
            epoch,
            model_lr,
            dodge,
            testStyle,
            uv_path,
            style_path,
            gen_dataset_directory,
            gen_researchers_directory,
            benchmark_arguments,
            specify_benchmark
        )

    @staticmethod
    def add_args(parser: ArgumentParser) -> None:

        parser.add_argument(
            "-dist",
            "--dist-weight",
            type=float,
            help="The distance weight used in the attack"
        )
        parser.add_argument(
            "-l1",
            "--l1-weight",
            type=float,
            help="The l1 weight used in the attack"
        )
        parser.add_argument(
            "-prc",
            "--percept-weight",
            type=float,
            help="The percept weight used in the attack"
        )
        parser.add_argument(
            "-tv",
            "--tv-weight",
            type=float,
            help="The TV weight used in the attack"
        )
        parser.add_argument(
            "-sty",
            "--style-weight",
            type=float,
            help="The style weight used in the attack"
        )
        parser.add_argument(
            "-ws",
            "--withStyle",
            type=str2Bool,
            help="Whether to optimize the style weights too"
        )
        parser.add_argument(
            "-thr",
            "--threshold",
            type=float,
            help="The threshold used in the attack"
        )
        parser.add_argument(
            "-ep",
            "--epoch",
            type=int,
            help="The max number of epochs used in the attack"
        )
        parser.add_argument(
            "-mlr",
            "--model-lr",
            type=float,
            help="The model learning rate used in the attack"
        )
        parser.add_argument(
            "-ddg",
            "--dodge",
            type=str2Bool,
            help="Whether to dodge or not"
        )
        parser.add_argument(
            "-ts",
            "--testStyle",
            type=str2Bool,
            help="Whether to test styling with less attacks"
        )
        parser.add_argument(
            "-uv",
            "--uv-path",
            type=str,
            help="The path that points to the uv mask."
        )
        parser.add_argument(
            "-style",
            "--style-path",
            type=str,
            help="The path that points to the style path."
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


# Hyperparameters for the flower interleave SAS benchmark (same used for the blue fire bench too)
class SASAccessoryGenerator(AccessoryGenerator):
    def __init__(self, args: SASBenchmarkArguments) -> None:
        self.args = args
        self.output_directory = args.benchmark_arguments.output_directory
        self.output_directory.mkdir(parents=True, exist_ok=True)
        self.output_directory = self.output_directory.as_posix()

        self.dist_weight = args.dist_weight 
        self.l1_weight = args.l1_weight
        self.percept_weight = args.percept_weight 
        self.tv_weight = args.tv_weight
        self.style_weight = args.style_weight
        self.withStyle = args.withStyle
        self.threshold = args.threshold
        self.epoch = args.epoch
        self.model_lr = args.model_lr
        self.uv = args.uv_path
        self.style = args.style_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.architecture = self.args.benchmark_arguments.architecture.construct(
            self.args.benchmark_arguments.dataset,
            self.args.benchmark_arguments.size,
            self.args.benchmark_arguments.weights_directory,
            device=self.device,
        )

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
        elif "mobilefacenet" in str(self.args.benchmark_arguments.weights_directory):
            rnetType = "mfn"
        
        
        datasetName = "PUBFIG"
        if self.args.benchmark_arguments.dataset == FaceDatasets.VGGFACE2:
            datasetName = "VGGFACE2"


        self.anchorBase = f"../anchors/masked_{rnetType}_{datasetName}_"


    def generate(
        self,
        gpu_manager: GPUDeviceManager,
        dataset: FaceDataset,
        size: FaceDatasetSize,
        base_class: int,
        target_class: Optional[int],
        base_class_name:str,
        target_class_name: Optional[str],
        universal_class_indices):

        logger = getLogger("SAS_benchmark")

        # names = IndividualDataset(...)
        # If self.faceInterleave is here, then we use all images of the person for one attack
        # If not, then we use a random image
        # note name set takes image names not anything else
    
        celebDir = self.args.gen_dataset_directory.as_posix()
        resDir = self.args.gen_researchers_directory.as_posix()
        all_classes = size.class_names

        if base_class < len(all_classes)-3:
            dir_path = celebDir
        else:
            dir_path = resDir

        dir_path += "/" + all_classes[base_class] + "/"
        gen_path = glob.glob(os.path.join(dir_path, '*.jpg')) + glob.glob(os.path.join(dir_path, '*.jpeg')) + glob.glob(os.path.join(dir_path, '*.JPG')) + glob.glob(os.path.join(dir_path, '*.JPEG'))
        gen_path = gen_path[:25]

        if target_class_name is None:
            anchorPic = f"{self.anchorBase}{base_class_name}.pth"
        else:
            anchorPic = f"{self.anchorBase}{target_class_name}.pth"
        
        adv_name = train_stymask(base_class, target_class, all_classes, self.architecture,
                    gen_path, self.uv, self.style, self.dist_weight, 
                    self.l1_weight, self.percept_weight, self.tv_weight, self.style_weight,
                    self.epoch, self.model_lr, self.output_directory, self.device, self.threshold, anchorPic, self.withStyle)

        return [Accessory(
            adv_name,
            self.style,
            base_class_name,
            target_class_name,
            base_class,
            target_class,
            universal_class_indices
        )]

class SASBenchmark(Component):
    @classmethod
    def run(cls, args: SASBenchmarkArguments) -> None:
        if args.dodge:
            class_names = ['GeorgeClooney', 'ReeseWitherspoon', 'HughGrant', 'DrewBarrymore', 'AngelinaJolie', 'ColinPowell', 'OrlandoBloom', 'JenniferLopez', 'KeiraKnightley', 'JenniferAniston', 'ReneeZellweger', 'GwynethPaltrow', 'JessicaSimpson', 'LeonardoDiCaprio', 'AliciaKeys', 'JodieFoster', 'TomCruise', 'MattDamon', 'NicoleKidman', 'CateBlanchett', 'SalmaHayek', 'EvaMendes', 'WillSmith', 'AvrilLavigne', 'CameronDiaz', 'JohnTravolta', 'DavidBeckham', 'CharlizeTheron', 'HalleBerry', 'BeyonceKnowles']
            if args.testStyle:
                class_names = ['DrewBarrymore', 'KeiraKnightley', 'HalleBerry', 'AngelinaJolie', 'DavidBeckham']

            accessory_generator = random_dodge_each(
                args.benchmark_arguments.dataset,
                args.benchmark_arguments.size,
                SASAccessoryGenerator(args),
                class_count=len(class_names),
                class_names=class_names
            )

        else:
            chosenBase =['n005219', 'n000266', 'n004216', 'n004538', 'n000176']
            chosenTargets = ['n001765', 'n004589', 'n004737', 'n002886', 'n007139', 'n001586']
            accessory_generator = random_pairwise_impersonation(
                args.benchmark_arguments.dataset,
                args.benchmark_arguments.size,
                SASAccessoryGenerator(args),
                class_count=6,
                class_names=chosenTargets,
                researchers_only=False,
                base_class_names = chosenBase
            )

        suite = BenchmarkSuite(args.specify_benchmark.benchmarks, accessory_generator)

        suite.run(args.benchmark_arguments)



if __name__ == "__main__":
    run_component(SASBenchmark, SASBenchmarkArguments, DEFAULT_GLOBAL_ARGUMENTS)
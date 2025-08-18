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
from advfaceutil.benchmark import pairwise_impersonation, dodge_each, random_dodge_each, random_pairwise_impersonation
from advfaceutil.benchmark import SpecifyBenchmarks
from advfaceutil.datasets import FaceDatasets
from advfaceutil.datasets import FaceDatasetSize
from advfaceutil.datasets import IndividualDataset
from advfaceutil.utils import Component
from advfaceutil.utils import ComponentArguments
from advfaceutil.utils import run_component
from advfaceutil.utils import DEFAULT_GLOBAL_ARGUMENTS

import numpy as np
from advfaceutil.datasets.faces.pubfig import PubFigDatasetSize

from Txt2Adv3DAttack import texture_attack
from DiffUtil import load_model_from_config
from omegaconf import OmegaConf
from torchvision.utils import save_image
from advfaceutil.recognition.processing import FaceProcessors, DlibFaceProcessor, MediaPipeFaceProcessor

DefaultedBenchmarkArguments = BenchmarkArguments.from_defaults(
    {
        BenchmarkArgument.ADDITIVE_OVERLAY: False,
        BenchmarkArgument.FACE_PROCESSOR: FaceProcessors.ADV_DIFF,
        BenchmarkArgument.OVERLAY_MODEL: Path("3Dobjs/Face_Mask/face_mask.obj")
    }
)

def str2Bool(s):
    return s.lower()=="true"
    

@dataclass(frozen=True)
class Text2Adv3DBenchmarkArguments(ComponentArguments):
    # May add change to control how we split the interleaving
    # Text2Adv Specific, defaults were for the flower interleave
    prompt: str #= "Pattern of flowers"
    seed: int #= 2003284
    s: float #= 5
    steps: float #= 0.5
    genSteps: int #= 100
    k: int #= 3
    a: int #= 20
    genScale: int #= 8
    randInitial: bool #= True
    innerK: int #= 3
    faceInterleave: int #= True
    nInter: int #= 1
    successThresh:float
    addLabel: bool
    matchLighting: bool
    dodge:str
    testStyle:bool
    anchorThresh:float
    embeddingAttack: bool
    config_path: Path
    diffusion_weights: Path
    # Dataset directory here is different than the ones in the general benchmark arguments
    # These ones are for generation, ones in general benchmark are for testing
    gen_dataset_directory: Path # Have to be unaligned
    gen_researchers_directory: Path # Have to be unaligned
    benchmark_arguments: BenchmarkArguments
    specify_benchmark: SpecifyBenchmarks

    @staticmethod
    def parse_args(args:Namespace) -> "Text2Adv3DBenchmarkArguments":
        prompt = args.prompt
        if not isinstance(prompt, str):
            raise Exception(
                f"The prompt must be a string, but '{prompt}' was given instead"
            )
        # Parsing the diffusion attack params
        seed = args.seed
        s = args.s
        steps = args.steps
        genSteps = args.genSteps
        k = args.k
        a = args.a
        genScale = args.genScale
        randInitial = args.randInitial
        innerK = args.innerK
        faceInterleave = args.faceInterleave
        nInter = args.nInter
        successThresh = args.successThresh
        addLabel = args.addLabel
        matchLighting = args.matchLighting
        dodge = args.dodge
        testStyle = args.testStyle
        anchorThresh = args.anchorThresh
        embeddingAttack = args.embeddingAttack
        if not isinstance(dodge, str):
            raise Exception(
                f"The dodge mode must be a string ('NONE', 'DIRECT', 'INDIRECT'), but '{dodge}' was given instead"
            )
        
        config_path = Path(args.config_path)
        if not config_path.exists() or config_path.is_dir():
            raise Exception(
                f"Config path ('{config_path})' provided is not a configuration file"
            )

        diffusion_weights = Path(args.diffusion_weights)
        if not diffusion_weights.exists() or diffusion_weights.is_dir():
            raise Exception(
                f"Diffusion weights ('{diffusion_weights})' provided are not weights"
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

        return  Text2Adv3DBenchmarkArguments(
            prompt,
            seed,
            s,
            steps,
            genSteps,
            k,
            a,
            genScale,
            randInitial,
            innerK,
            faceInterleave,
            nInter,
            successThresh,
            addLabel,
            matchLighting,
            dodge,
            testStyle,
            anchorThresh,
            embeddingAttack,
            config_path,
            diffusion_weights,
            gen_dataset_directory,
            gen_researchers_directory,
            benchmark_arguments,
            specify_benchmark
        )

    @staticmethod
    def add_args(parser: ArgumentParser) -> None:
        parser.add_argument(
            "-prmt",
            "--prompt", 
            type=str,
            help="The prompt used by the diffusion model"
        )
        parser.add_argument(
            "-sd",
            "--seed",
            type=int,
            help="The seed used in the diffusion model"
        )
        parser.add_argument(
            "-s",
            "--s",
            type=float,
            help="The step size of the adversarial update"
        )
        parser.add_argument(
            "-stps",
            "--steps",
            type=float,
            help="The proportion of the sampling that is adversarial, i.e 0.5 means that half the adversarial phase starts halfway through generation"
        )
        parser.add_argument(
            "-gsteps",
            "--genSteps",
            type=int,
            help="The number of steps used in the sampling"
        )
        parser.add_argument(
            "-k",
            "--k",
            type=int,
            help="The number of iterations on the outmost loop"
        )
        parser.add_argument(
            "-a",
            "--a",
            type=int,
            help="The step size used on the adversarial update between iterations of the outmost loop"
        )
        parser.add_argument(
            "-gscale",
            "--genScale",
            type=int,
            help="The value of the generative scale used in the diffusion model"
        )
        parser.add_argument(
            "-rinit",
            "--randInitial",
            type=str2Bool,
            help="If True, then random noise is used initially as a latent otherwise a better initial latent is used"
        )
        parser.add_argument(
            "-ik",
            "--innerK",
            type=int,
            help="The number of adversarial update steps the attack uses per adversarial phase"
        )
        parser.add_argument(
            "-fLeave",
            "--faceInterleave",
            type=int,
            help="The number of images used during generation"
        )
        parser.add_argument(
            "-nI",
            "--nInter",
            type=int,
            help="The number of images used to make each adversarial update step"
        )
        parser.add_argument(
            "-sT",
            "--successThresh",
            type=float,
            help="The proportion of the images that we need the attack to be successful with before it ends"
        )
        parser.add_argument(
            "-aL",
            "--addLabel",
            type=str2Bool,
            help="Whether the label is used during generation"
        )
        parser.add_argument(
            "-mL",
            "--matchLighting",
            type=str2Bool,
            help="Whether to match lighting during generation"
        )
        parser.add_argument(
            "-ddg",
            "--dodge",
            type=str,
            help="'NONE' when doing impersonation, 'DIRECT' when doing direct dodging and 'INDIRECT' when doing indirect dodging"
        )
        parser.add_argument(
            "-ts",
            "--testStyle",
            type=str2Bool,
            help="Whether to test styling with less attacks"
        )
        parser.add_argument(
            "-aT",
            "--anchorThresh",
            type=float,
            help="The threshold used in the embedding attack to record a successful attack"
        )
        parser.add_argument(
            "-ea",
            "--embeddingAttack",
            type=str2Bool,
            help="Whether to perform an embedding attack during generation"
        )
        parser.add_argument(
            "-conf",
            "--config-path",
            type=str,
            help="The path that points to the config file"
        )
        parser.add_argument(
            "-diff",
            "--diffusion-weights",
            type=str,
            help="The path that points to where the weights of the diffusion model are stored"
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


# Hyperparameters for the flower interleave Text2Adv3D benchmark (same used for the blue fire bench too)
class Text2Adv3DAccessoryGenerator(AccessoryGenerator):
    def __init__(self, args: Text2Adv3DBenchmarkArguments) -> None:
        self.args = args
        self.output_directory = args.benchmark_arguments.output_directory
        self.output_directory.mkdir(parents=True, exist_ok=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.architecture = self.args.benchmark_arguments.architecture.construct(
            self.args.benchmark_arguments.dataset,
            self.args.benchmark_arguments.size,
            self.args.benchmark_arguments.weights_directory,
            self.device)

        self.config_path = args.config_path
        self.diffusion_weights = args.diffusion_weights
    
        self.prompt = args.prompt
        self.seed = args.seed
        self.s = args.s
        self.steps = args.steps
        self.genSteps = args.genSteps
        self.k = args.k
        self.a = args.a
        self.genScale = args.genScale
        self.randInitial = args.randInitial
        self.innerK = args.innerK
        self.faceInterleave = args.faceInterleave
        self.nInter = args.nInter
        self.successThresh = args.successThresh
        self.addLabel = args.addLabel
        self.matchLighting = args.matchLighting
        self.dodgeMode = args.dodge == "INDIRECT"
        self.anchorThresh = args.anchorThresh
        self.embeddingAttack = args.embeddingAttack

        if self.embeddingAttack:
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


            self.anchorBase = f"anchors/masked_{rnetType}_{datasetName}_"

    @cached_property
    def model(self):
        return load_model_from_config(OmegaConf.load("configs/stable-diffusion/v2-inference.yaml"),
            "diffusion_weights",
            torch.device("cuda" if torch.cuda.is_available() else "cpu"))

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

        logger = getLogger("Text2Adv3D_benchmark")

        logger.info("Getting GPU")

        # device = gpu_manager.acquire()
        logger.info("Loading network")
        logger.info("Architecture loaded")

        celebDir = self.args.gen_dataset_directory.as_posix()
        resDir = self.args.gen_researchers_directory.as_posix()
        baseClassName = size.class_names[base_class]

        anchorPic = ""
        if self.embeddingAttack:
            if target_class_name is None:
                anchorPic = f"{self.anchorBase}{base_class_name}.pth"
            else:
                anchorPic = f"{self.anchorBase}{target_class_name}.pth"

            logger.info(f"Using anchor of {anchorPic}")

        logger.info("Non Adv Being Created")
        # Non adversarial generation first, negative steps so that no adv is ever done
        non_adv_accessory_path = texture_attack(prompt=self.prompt, name_set=[], desc="NonAdv_",
                                clfSet=[self.architecture], anchorThresh=self.anchorThresh, anchorPic=anchorPic,
                                img_class=base_class, target_class=target_class, s=self.s, steps=-1,
                                seed=self.seed, scaled=False, k=1, genS=self.genSteps, a=self.a, scale=self.genScale,
                                randInitial=self.randInitial, innerK=self.innerK, device=self.device, toSave=False,
                                outpath=self.output_directory.as_posix()+"/", model=self.model,
                                faceClasses=size.class_names, nInter=self.nInter, logger=logger)

        logger.info("Adv Being Created")
        image_extensions = [".jpg", ".jpeg", ".png"]
        picNames = []
        if os.path.exists(celebDir+"/"+baseClassName):
                for root, dirs, files in os.walk(celebDir+"/"+baseClassName):
                    for filename in files:
                        if any(filename.lower().endswith(ext) for ext in image_extensions):
                            picNames.append(os.path.join(root, filename))
        elif os.path.exists(resDir+"/"+baseClassName):
            for root, dirs, files in os.walk(resDir+"/"+baseClassName):
                for filename in files:
                    if any(filename.lower().endswith(ext) for ext in image_extensions):
                        picNames.append(os.path.join(root, filename))
        
        mesh_path="3Dobjs/Face_Mask/face_mask.obj"
        tmp_mesh_path="3Dobjs/Face_Mask/tmp.obj"
        
        adv_accessory_path = texture_attack(prompt=self.prompt, name_set=picNames[:min(len(picNames), self.faceInterleave)],
                            clfSet=[self.architecture], desc="Adv_", addLabel = self.addLabel, matchLighting=self.matchLighting, indirectDodge=self.dodgeMode,
                            img_class=base_class, target_class=target_class, s=self.s, steps=self.steps, successThresh=self.successThresh,
                            seed=self.seed, scaled=False, k=self.k, genS=self.genSteps, a=self.a, scale=self.genScale,
                            randInitial=self.randInitial, innerK=self.innerK, device=self.device, toSave=False, anchorPic=anchorPic,
                            outpath=self.output_directory.as_posix()+"/", anchorThresh=self.anchorThresh, model=self.model,
                            faceClasses=size.class_names, nInter=self.nInter, logger=logger,
                            mesh_path=mesh_path, tmp_mesh_path=tmp_mesh_path)
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

class Text2Adv3DBenchmark(Component):
    @classmethod
    def run(cls, args: Text2Adv3DBenchmarkArguments) -> None:
        if args.dodge != "NONE":
            class_names = ['GeorgeClooney', 'ReeseWitherspoon', 'HughGrant', 'DrewBarrymore', 'AngelinaJolie', 'ColinPowell', 'OrlandoBloom', 'JenniferLopez', 'KeiraKnightley', 'JenniferAniston', 'ReneeZellweger', 'GwynethPaltrow', 'JessicaSimpson', 'LeonardoDiCaprio', 'AliciaKeys', 'JodieFoster', 'TomCruise', 'MattDamon', 'NicoleKidman', 'CateBlanchett', 'SalmaHayek', 'EvaMendes', 'WillSmith', 'AvrilLavigne', 'CameronDiaz', 'JohnTravolta', 'DavidBeckham', 'CharlizeTheron', 'HalleBerry', 'BeyonceKnowles']
            if args.testStyle:
                class_names = ['DrewBarrymore', 'KeiraKnightley', 'HalleBerry', 'AngelinaJolie', 'DavidBeckham']
                
            accessory_generator = random_dodge_each(
                args.benchmark_arguments.dataset,
                args.benchmark_arguments.size,
                Text2Adv3DAccessoryGenerator(args),
                class_count=len(class_names),
                class_names=class_names
            )
        else:
            chosenBase =['n005219', 'n000266', 'n004216', 'n004538', 'n000176']
            chosenTargets = ['n001765', 'n004589', 'n004737', 'n002886', 'n007139', 'n001586']
            accessory_generator = random_pairwise_impersonation(
                args.benchmark_arguments.dataset,
                args.benchmark_arguments.size,
                Text2Adv3DAccessoryGenerator(args),
                class_count=6,
                class_names=chosenTargets,
                researchers_only=False,
                base_class_names = chosenBase
            )

        suite = BenchmarkSuite(args.specify_benchmark.benchmarks, accessory_generator)

        suite.run(args.benchmark_arguments)



if __name__ == "__main__":
    run_component(Text2Adv3DBenchmark, Text2Adv3DBenchmarkArguments, DEFAULT_GLOBAL_ARGUMENTS)
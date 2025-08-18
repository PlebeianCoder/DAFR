from abc import ABCMeta
from argparse import ArgumentParser
from argparse import Namespace
from dataclasses import dataclass
from logging import getLogger
from pathlib import Path
from random import uniform
from typing import Callable

import numpy as np
import torch
from torchvision.transforms import transforms

import wandb
from advfaceutil.datasets import FaceDatasets
from advfaceutil.datasets import FaceDatasetSize
from advfaceutil.recognition.architectures import RecognitionArchitectures
from advfaceutil.recognition.base import RecognitionArchitecture
from advfaceutil.recognition.occlusion import Occlusion
from advfaceutil.utils import Component
from advfaceutil.utils import ComponentArguments

LOGGER = getLogger("training")


@dataclass(frozen=True)
class RecognitionTrainingArguments(ComponentArguments):
    training_dataset_directory: Path
    training_researchers_directory: Path
    testing_dataset_directory: Path
    testing_researchers_directory: Path
    weights_directory: Path
    output_directory: Path
    dataset: FaceDatasets
    size: FaceDatasetSize
    training_image_limit: int = 40
    testing_image_limit: int = 5

    num_epochs: int = 100
    batch_size: int = 4
    learning_rate: float = 0.00005

    @staticmethod
    def parse_args(args: Namespace) -> "RecognitionTrainingArguments":
        training_dataset_directory = Path(args.training_dataset_directory)
        if (
            not training_dataset_directory.exists()
            or not training_dataset_directory.is_dir()
        ):
            raise Exception(
                f" Training dataset directory must be a valid directory but was given {training_dataset_directory}"
            )
        training_researchers_directory = Path(args.training_researchers_directory)
        if (
            not training_researchers_directory.exists()
            or not training_researchers_directory.is_dir()
        ):
            raise Exception(
                f"Training researchers directory must be a valid directory but was given {training_researchers_directory}"
            )
        testing_dataset_directory = Path(args.testing_dataset_directory)
        if (
            not testing_dataset_directory.exists()
            or not testing_dataset_directory.is_dir()
        ):
            raise Exception(
                f" Testing dataset directory must be a valid directory but was given {testing_dataset_directory}"
            )
        testing_researchers_directory = Path(args.testing_researchers_directory)
        if (
            not testing_researchers_directory.exists()
            or not testing_researchers_directory.is_dir()
        ):
            raise Exception(
                f"Testing researchers directory must be a valid directory but was given {testing_researchers_directory}"
            )

        weights_directory = Path(args.weights_directory)
        output_directory = Path(args.output_directory)
        dataset = FaceDatasets[args.dataset]
        size = dataset.get_size(args.size)
        training_image_limit = args.training_image_limit
        testing_image_limit = args.testing_image_limit
        epochs = args.epochs
        batch_size = args.batch_size
        learning_rate = args.learning_rate

        return RecognitionTrainingArguments(
            training_dataset_directory,
            training_researchers_directory,
            testing_dataset_directory,
            testing_researchers_directory,
            weights_directory,
            output_directory,
            dataset,
            size,
            training_image_limit,
            testing_image_limit,
            epochs,
            batch_size,
            learning_rate,
        )

    @staticmethod
    def add_args(parser: ArgumentParser) -> None:
        parser.add_argument(
            "training_dataset_directory",
            type=str,
            help="The training directory for dataset images",
        )
        parser.add_argument(
            "training_researchers_directory",
            type=str,
            help="The training directory for the researchers images",
        )
        parser.add_argument(
            "testing_dataset_directory",
            type=str,
            help="The testing directory for dataset images",
        )
        parser.add_argument(
            "testing_researchers_directory",
            type=str,
            help="The testing directory for the researchers images",
        )
        parser.add_argument(
            "weights_directory",
            type=str,
            help="The path to the weights directory that includes the base model weights",
        )
        parser.add_argument("output_directory", type=str, help="The output directory")
        parser.add_argument(
            "dataset",
            type=str,
            choices=[d.name for d in FaceDatasets],
            help="The dataset to use",
        )
        parser.add_argument(
            "size",
            type=str,
            choices=["SMALL", "LARGE"],
            help="The size of the dataset to use",
        )
        parser.add_argument(
            "-train",
            "--training-image-limit",
            type=int,
            default=40,
            help="The number of training images for each class",
        )
        parser.add_argument(
            "-test",
            "--testing-image-limit",
            type=int,
            default=5,
            help="The number of testing images for each class",
        )
        parser.add_argument(
            "-e",
            "--epochs",
            type=int,
            default=100,
            help="The number of epochs to train for",
        )
        parser.add_argument(
            "-bs",
            "--batch-size",
            type=int,
            default=4,
            help="The batch size for each forward pass",
        )
        parser.add_argument(
            "-lr",
            "--learning-rate",
            type=float,
            default=0.00005,
            help="The learning rate",
        )


class RandomGaussianNoise(torch.nn.Module):
    def __init__(self, *, p: float, std: float, mean: float):
        super().__init__()
        self.p = p
        self.std = std
        self.mean = mean

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if uniform(0, 1) <= self.p:
            gaussian_noise = torch.randn(image.size()) * self.std + self.mean
            image = torch.clamp(image + gaussian_noise, 0, 1)

        return image


class TrainingComponent(Component, metaclass=ABCMeta):
    @staticmethod
    def _construct_augmentation() -> Callable:
        return transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                Occlusion(),
                RandomGaussianNoise(p=0.5, std=0.1, mean=0),
            ]
        )

    @staticmethod
    def _network_forward_pass(
        network: RecognitionArchitecture, inputs: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        return network(inputs)

    @classmethod
    def _train(
        cls,
        args: RecognitionTrainingArguments,
        architecture: RecognitionArchitectures,
        num_epochs: int,
        batch_size: int,
        learning_rate: float,
    ):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        classes = args.size.classes

        network_name = architecture.name.lower()
        network = architecture.construct(
            args.dataset,
            args.size,
            args.weights_directory,
            training=True,
            device=device,
        )

        size_name = args.size.name.lower()

        LOGGER.info("%s %s loaded", network_name.title(), size_name)

        criterion = torch.nn.CrossEntropyLoss()
        optimiser = torch.optim.Adam(network.parameters(), lr=learning_rate)

        # Construct the data loaders
        training_loader = args.dataset.construct(
            args.training_dataset_directory,
            args.training_researchers_directory,
            args.size,
            args.training_image_limit,
        ).as_loader(batch_size, shuffle=True)

        testing_loader = args.dataset.construct(
            args.testing_dataset_directory,
            args.testing_researchers_directory,
            args.size,
            args.testing_image_limit,
        ).as_loader(1, shuffle=False)

        # Construct the augmentation that we use during training
        training_augmentation = cls._construct_augmentation()

        wandb.init(
            project=f"{network_name}-training",
            entity="adversaries",
            config={
                "learning_rate": learning_rate,
                "epochs": num_epochs,
                "batch_size": batch_size,
                "size": args.size.name,
                "classes": args.size.classes,
                "training_image_limit": args.training_image_limit,
                "testing_image_limit": args.testing_image_limit,
                "architecture": network_name,
                "dataset": args.dataset.name,
            },
        )

        best_epoch, best_accuracy, best_loss = -1, 0, float("inf")

        args.output_directory.mkdir(parents=True, exist_ok=True)

        # Ensure that the full array is printed
        np.printoptions(threshold=np.int64.size)

        for epoch in range(num_epochs):
            network.train()
            epoch_loss = 0
            # Training loop
            LOGGER.info(f"Beginning epoch {epoch}")

            for inputs, labels in training_loader:
                # To remove the previous gradients
                optimiser.zero_grad()

                # Use augmentation
                inputs = training_augmentation(inputs)

                # Move the data to the device
                inputs = inputs.to(device)
                labels = labels.to(device)

                final_outputs = cls._network_forward_pass(network, inputs, labels)

                # Calculate the loss
                loss = criterion(final_outputs, labels)

                # Update the epoch loss
                epoch_loss += loss.item()

                # Backpropagation
                loss.backward()
                optimiser.step()

            # Validate with test
            LOGGER.info(f"Testing epoch {epoch}")
            with torch.no_grad():
                network.eval()
                correct, samples = 0, 0

                for inputs, labels in testing_loader:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    outputs = network(inputs)
                    for i in range(outputs.size()[0]):
                        _, classification = torch.max(outputs[i], 0)
                        samples += 1
                        # Image metrics
                        if labels[i][classification].item() == 1.0:
                            correct += 1

                # Save a checkpoint if the accuracy is better or if the accuracy is the same and the loss is better
                if correct / samples > best_accuracy or (
                    correct / samples >= best_accuracy and epoch_loss < best_loss
                ):
                    best_epoch, best_accuracy, best_loss = (
                        epoch,
                        correct / samples,
                        epoch_loss,
                    )
                    LOGGER.info(
                        f"Best Epoch {best_epoch}: {correct}/{samples} with {best_accuracy * 100}%"
                    )

                    save_directory = (
                        args.output_directory
                        / f"{network_name}-{args.dataset.name.lower()}-{args.size.name.lower()}-{epoch}"
                    )
                    save_directory.mkdir(parents=True, exist_ok=True)

                    network.save_transfer_data(
                        save_directory,
                        args.dataset,
                        args.size,
                    )
                elif args.size.is_large and uniform(0, 1) >= 0.9:
                    network.load_transfer_data(
                        args.output_directory
                        / f"{network_name}-{args.dataset.name.lower()}-{args.size.name.lower()}-{best_epoch}",
                        args.dataset,
                        args.size,
                        device,
                    )
                    LOGGER.info(f"Resetting back to {best_epoch}")

                    # Reset the optimiser due to reload
                    optimiser = torch.optim.Adam(network.parameters(), lr=learning_rate)

            LOGGER.info(
                f"Epoch {epoch} Loss: {epoch_loss} Accuracy: {correct / samples}"
            )
            wandb.log(
                {"epoch": epoch, "loss": epoch_loss, "accuracy": correct / samples}
            )

        # Create an artifact that contains the weights
        model_artifact = wandb.Artifact(
            network_name, type="model", metadata=dict(wandb.config)
        )

        model_artifact.add_dir(
            (
                args.output_directory
                / f"{network_name}-{args.dataset.name.lower()}-{args.size.name.lower()}-{best_epoch}"
            ).as_posix()
        )

        wandb.log_artifact(model_artifact)

        wandb.finish()

        LOGGER.info(
            "%s %s training complete with accuracy %.2f",
            network_name.title(),
            size_name,
            best_accuracy,
        )

from argparse import ArgumentParser
from argparse import Namespace
from dataclasses import dataclass

from advfaceutil.recognition.architectures import RecognitionArchitectures
from advfaceutil.recognition.training import RecognitionTrainingArguments
from advfaceutil.recognition.training import TrainingComponent


@dataclass(frozen=True)
class ClipTrainingArguments(RecognitionTrainingArguments):
    model_name: str = "FaRL"

    @staticmethod
    def parse_args(args: Namespace) -> "ClipTrainingArguments":
        base_args = super().parse_args(args)

        model_name = args.model_name

        return ClipTrainingArguments(
            base_args.training_dataset_directory,
            base_args.training_researchers_directory,
            base_args.testing_dataset_directory,
            base_args.testing_researchers_directory,
            base_args.weights_directory,
            base_args.output_directory,
            base_args.dataset,
            base_args.size,
            base_args.training_image_limit,
            base_args.testing_image_limit,
            base_args.num_epochs,
            base_args.batch_size,
            base_args.learning_rate,
            model_name,
        )

    @staticmethod
    def add_args(parser: ArgumentParser) -> None:
        super().add_args(parser)
        parser.add_argument(
            "--model-name",
            type=str,
            default="FaRL",
            help="The name of the CLIP model to train.",
        )


class ClipTraining(TrainingComponent):
    @staticmethod
    def run(args: ClipTrainingArguments) -> None:
        if args.model_name.lower() == "farl":
            architecture = RecognitionArchitectures.FARL
        else:
            raise ValueError("Unsupported model %s" % args.model_name)

        ClipTraining._train(
            args,
            architecture,
            args.num_epochs,
            args.batch_size,
            args.learning_rate,
        )

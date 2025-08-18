import torch

from advfaceutil.recognition.architectures import RecognitionArchitectures
from advfaceutil.recognition.base import RecognitionArchitecture
from advfaceutil.recognition.training import RecognitionTrainingArguments
from advfaceutil.recognition.training import TrainingComponent


class IResNetHeadTraining(TrainingComponent):
    @staticmethod
    def run(args: RecognitionTrainingArguments) -> None:
        IResNetHeadTraining._train(
            args,
            RecognitionArchitectures.IRESNETHEAD,
            args.num_epochs,
            args.batch_size,
            args.learning_rate,
        )

    @staticmethod
    def _network_forward_pass(
        network: RecognitionArchitecture, inputs: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        return network(inputs, labels)

from advfaceutil.recognition.architectures import RecognitionArchitectures
from advfaceutil.recognition.base import RecognitionArchitecture
from advfaceutil.recognition.training import RecognitionTrainingArguments
from advfaceutil.recognition.training import TrainingComponent


class IResNetTraining(TrainingComponent):
    @staticmethod
    def run(args: RecognitionTrainingArguments) -> None:
        IResNetTraining._train(
            args,
            RecognitionArchitectures.IRESNET,
            args.num_epochs,
            args.batch_size,
            args.learning_rate,
        )

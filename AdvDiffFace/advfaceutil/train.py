from advfaceutil.recognition.insightface.training import IResNetTraining
from advfaceutil.recognition.iresnethead.training import IResNetHeadTraining
from advfaceutil.recognition.training import RecognitionTrainingArguments
from advfaceutil.utils import ComponentEnum
from advfaceutil.utils import DEFAULT_GLOBAL_ARGUMENTS
from advfaceutil.utils import run


class TrainingComponents(ComponentEnum):
    IRESNET = (
        RecognitionTrainingArguments,
        IResNetTraining,
    )
    IRESNETHEAD = (
        RecognitionTrainingArguments,
        IResNetHeadTraining,
    )


if __name__ == "__main__":
    run(TrainingComponents, DEFAULT_GLOBAL_ARGUMENTS)

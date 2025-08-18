__all__ = ["Clip", "FaRL"]

from abc import ABCMeta
from pathlib import Path
from typing import Optional

from PIL import Image
import numpy as np

from advfaceutil.datasets import FaceDatasets, FaceDatasetSize
from advfaceutil.recognition.base import RecognitionArchitecture
import torchvision

import torch
import clip

from advfaceutil.utils import validate_files_existence


class Clip(RecognitionArchitecture, metaclass=ABCMeta):
    crop_size = 112
    # For practical reasons

    def __init__(self, name: str) -> None:
        super().__init__()
        self._name = name

        self._clip, _ = clip.load(name)

    def load_transfer_data(
        self,
        weights_directory: Path,
        dataset: FaceDatasets,
        size: FaceDatasetSize,
        device: torch.device,
    ) -> None:
        backbone_path = weights_directory / f"{self.name}-backbone.pth"

        validate_files_existence(backbone_path)

        state = torch.load(backbone_path, device)
        self._clip.load_state_dict(state["state_dict"], strict=False)
        self._clip = self._clip.to(device)

    def save_transfer_data(
        self, save_directory: Path, dataset: FaceDatasets, size: FaceDatasetSize
    ) -> None:
        save_directory.mkdir(exist_ok=True)

        torch.save(
            self._clip.state_dict(),
            save_directory / f"{self.name}-backbone.pth",
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        # expect (1, 3, H, W)
        if image.size()[image.dim() - 1] != 224:
            resizeOp = torchvision.transforms.Resize(224)
            reImage = resizeOp(image)
        else:
            reImage = image
        # # expects channel last
        # base = np.transpose(image.cpu().numpy()[0], (1, 2, 0))

        # base = (base * 255).astype(np.uint8)
        # base = Image.fromarray(base)

        # processed_image = self._preprocess(base).unsqueeze(0)
        # processed_image = processed_image.to(next(self._clip.parameters()).device)

        normalize = torchvision.transforms.Compose(
            [
                torchvision.transforms.Normalize(
                    (0.48145466, 0.4578275, 0.40821073),
                    (0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )

        return self._clip.encode_image(normalize(reImage))

    def returnEmbedding(self, image):
        return self.forward(image)


class FaRL(Clip):
    name = "FaRL"
    noBGR = True
    batched = True

    @staticmethod
    def construct(
        dataset: FaceDatasets,
        size: FaceDatasetSize,
        weights_directory: Optional[Path] = None,
        training: bool = False,
        device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
    ) -> "FaRL":
        model = FaRL()
        if weights_directory is not None:
            model.load_transfer_data(weights_directory, dataset, size, device)

        model.to(device)

        if not training:
            model.eval()

        return model

    def __init__(self) -> None:
        self.noBGR = True
        self.batched = True
        super().__init__("ViT-B/16")

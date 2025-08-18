__all__ = ["SSIM"]

from typing import Any

from skimage.metrics import structural_similarity as ssim
import cv2

from advfaceutil.benchmark.data import Accessory, DataPropertyEnum, DataBin
from advfaceutil.benchmark.statistic.base import AccessoryStatistic


class SSIMProperties(DataPropertyEnum):
    SSIM = "ssim"


class SSIM(AccessoryStatistic):
    @staticmethod
    def name() -> str:
        return "SSIM"

    def record_statistic(
        self,
        accessory: Accessory,
    ) -> None:
        if (
            accessory.non_adversarial_accessory.shape[0]
            != accessory.adversarial_accessory.shape[0]
            or accessory.non_adversarial_accessory.shape[0]
            != accessory.adversarial_accessory.shape[0]
        ):
            t = cv2.resize(
                accessory.non_adversarial_accessory,
                (
                    accessory.adversarial_accessory.shape[0],
                    accessory.adversarial_accessory.shape[1],
                ),
                interpolation=cv2.INTER_CUBIC,
            )
        else:
            t = accessory.non_adversarial_accessory
        accessory.add_property(
            SSIMProperties.SSIM,
            ssim(
                t,
                accessory.adversarial_accessory,
                multichannel=True,
                data_range=1.0,
                channel_axis=2,
            ),
        )

    def collate_statistics(self, data_bin: DataBin[Accessory]) -> Any:
        return self._collate_list_statistics(data_bin, SSIMProperties.SSIM)

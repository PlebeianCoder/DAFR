__all__ = ["PSNR"]

from typing import Any

from skimage.metrics import peak_signal_noise_ratio as psnr
import cv2

from advfaceutil.benchmark.data import DataBin
from advfaceutil.benchmark.data import DataPropertyEnum
from advfaceutil.benchmark.data import Accessory
from advfaceutil.benchmark.statistic.base import AccessoryStatistic


class PSNRProperties(DataPropertyEnum):
    PSNR = "psnr"


class PSNR(AccessoryStatistic):
    @staticmethod
    def name() -> str:
        return "PSNR"

    def record_statistic(self, accessory: Accessory) -> None:
        # Just to check shape
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
            PSNRProperties.PSNR,
            psnr(
                t,
                accessory.adversarial_accessory,
                data_range=1,
            ),
        )

    def collate_statistics(self, data_bin: DataBin[Accessory]) -> Any:
        return self._collate_list_statistics(data_bin, PSNRProperties.PSNR)

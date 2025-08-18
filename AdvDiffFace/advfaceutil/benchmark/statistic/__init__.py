__all__ = [
    "Statistic",
    "StatisticFactory",
    "collate_statistics_for_bin",
    "AccessoryStatistic",
    "AdversarialGap",
    "AdversarialGapProperties",
    "AngleVariance",
    "AngleVarianceProperties",
    "CMMD",
    "PSNR",
    "SSIM",
    "SuccessRate",
    "SuccessRateProperties",
    "TopK",
    "TopKProperties",
    "DEFAULT_STATISTICS",
    "DEFAULT_ACCESSORY_STATISTICS",
]

from advfaceutil.benchmark.statistic.base import (
    Statistic,
    StatisticFactory,
    collate_statistics_for_bin,
    AccessoryStatistic,
)
from advfaceutil.benchmark.statistic.adversarial_gap import (
    AdversarialGap,
    AdversarialGapProperties,
)
from advfaceutil.benchmark.statistic.angle import AngleVariance, AngleVarianceProperties
from advfaceutil.benchmark.statistic.cmmd import CMMD
from advfaceutil.benchmark.statistic.psnr import PSNR
from advfaceutil.benchmark.statistic.ssim import SSIM
from advfaceutil.benchmark.statistic.success import SuccessRate, SuccessRateProperties
from advfaceutil.benchmark.statistic.cosineSuccess import (
    CosineSuccessRate,
    CosineSuccessRateProperties,
)
from advfaceutil.benchmark.statistic.cosineStat import CosineStat, CosineStatProperties
from advfaceutil.benchmark.statistic.embedSuccess import (
    EmbedSuccessRate,
    EmbedSuccessRateProperties,
)
from advfaceutil.benchmark.statistic.topk import TopK, TopKProperties


DEFAULT_STATISTICS = [
    # AdversarialGap.Factory(),
    AngleVariance.Factory(),
    CosineSuccessRate.Factory(),
    CosineStat.Factory()
    # SuccessRate.Factory(),
    # TopK.Factory(3),
]

DEFAULT_ACCESSORY_STATISTICS = [
    PSNR(),
    SSIM(),
    CMMD(),
]

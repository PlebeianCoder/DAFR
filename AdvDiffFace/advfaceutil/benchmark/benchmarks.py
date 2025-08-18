__all__ = [
    "DEFAULT_BENCHMARKS",
    "DEFAULT_BIN_PROPERTY_COMBINATIONS",
    "BaselineBenchmarkFactory",
    "MatchLightingBenchmarkFactory",
    "BrightnessVarianceBenchmarkFactory",
]

from advfaceutil.benchmark.augmentation import MatchLighting
from advfaceutil.benchmark.augmentation import UniversalClasses
from advfaceutil.benchmark.augmentation import UniversalClassesProperties
from advfaceutil.benchmark.augmentation import VaryLighting
from advfaceutil.benchmark.augmentation import VaryLightingProperties
from advfaceutil.benchmark.base import Benchmark
from advfaceutil.benchmark.statistic import AngleVarianceProperties
from advfaceutil.benchmark.statistic import DEFAULT_ACCESSORY_STATISTICS
from advfaceutil.benchmark.statistic import DEFAULT_STATISTICS


DEFAULT_BIN_PROPERTY_COMBINATIONS = [
    (AngleVarianceProperties.PITCH.with_sized_bin(10),),
    (AngleVarianceProperties.YAW.with_sized_bin(10),),
    (
        AngleVarianceProperties.PITCH.with_sized_bin(10),
        AngleVarianceProperties.YAW.with_sized_bin(10),
    ),
    (UniversalClassesProperties.UNIVERSAL_EXAMPLE,),
    (UniversalClassesProperties.CLASS_NAME,),
    (
        UniversalClassesProperties.CLASS_NAME,
        UniversalClassesProperties.UNIVERSAL_EXAMPLE,
    ),
    (
        AngleVarianceProperties.PITCH.with_sized_bin(10),
        AngleVarianceProperties.YAW.with_sized_bin(10),
        UniversalClassesProperties.UNIVERSAL_EXAMPLE,
    ),
]


class BaselineBenchmarkFactory(Benchmark.Factory):
    def __init__(self):
        super().__init__(
            [UniversalClasses.Factory()],
            DEFAULT_STATISTICS,
            DEFAULT_ACCESSORY_STATISTICS,
            DEFAULT_BIN_PROPERTY_COMBINATIONS,
        )

    @staticmethod
    def name() -> str:
        return "Baseline"


class MatchLightingBenchmarkFactory(Benchmark.Factory):
    def __init__(self):
        super().__init__(
            [UniversalClasses.Factory(), MatchLighting.Factory()],
            DEFAULT_STATISTICS,
            DEFAULT_ACCESSORY_STATISTICS,
            DEFAULT_BIN_PROPERTY_COMBINATIONS,
        )

    @staticmethod
    def name() -> str:
        return "MatchLighting"


class BrightnessVarianceBenchmarkFactory(Benchmark.Factory):
    def __init__(self):
        super().__init__(
            [UniversalClasses.Factory(), VaryLighting.Factory(contrast_range=(0, 0))],
            DEFAULT_STATISTICS,
            DEFAULT_ACCESSORY_STATISTICS,
            DEFAULT_BIN_PROPERTY_COMBINATIONS
            + [(VaryLightingProperties.BRIGHTNESS_CHANGE.with_sized_bin(0.5),)],
        )

    @staticmethod
    def name() -> str:
        return "BrightnessVariance"


DEFAULT_BENCHMARKS = [
    BaselineBenchmarkFactory(),
    MatchLightingBenchmarkFactory(),
    BrightnessVarianceBenchmarkFactory(),
]

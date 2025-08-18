__all__ = ["AngleVariance", "AngleVarianceProperties"]

from typing import Any
from functools import lru_cache

import cv2
import numpy as np
from mediapipe import solutions
from pyfacear import (
    Environment,
    OriginPointLocation,
    PerspectiveCamera,
    landmarks_from_results,
    FaceGeometry,
)

from advfaceutil.benchmark.args import BenchmarkArguments
from advfaceutil.benchmark.data import (
    BenchmarkData,
    DataPropertyEnum,
    DataBin,
    Accessory,
)
from advfaceutil.benchmark.statistic.base import Statistic, StatisticFactory


class AngleVarianceProperties(DataPropertyEnum):
    PITCH = "angle_pitch"
    YAW = "angle_yaw"
    ROLL = "angle_roll"


@lru_cache(maxsize=1)
def get_face_mesh() -> solutions.face_mesh.FaceMesh:
    return solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)


class AngleVariance(Statistic):
    class Factory(StatisticFactory):
        def __init__(
            self,
            environment: Environment = Environment(
                origin_point_location=OriginPointLocation.BOTTOM_LEFT_CORNER,
                perspective_camera=PerspectiveCamera(),
            ),
        ):
            self.environment = environment

        @staticmethod
        def name() -> str:
            return "AngleVariance"

        def construct(
            self,
            benchmark_arguments: BenchmarkArguments,
            accessory: Accessory,
        ) -> "AngleVariance":
            return AngleVariance(
                self.name(), benchmark_arguments, accessory, self.environment
            )

    def __init__(
        self,
        name: str,
        benchmark_arguments: BenchmarkArguments,
        accessory: Accessory,
        environment: Environment,
    ):
        super().__init__(name, benchmark_arguments, accessory)
        self.environment = environment

    def __eq__(self, other: Any) -> bool:
        return (
            isinstance(other, AngleVariance) and self.environment == other.environment
        )

    def __hash__(self) -> int:
        return hash((self.name, self.environment))

    def record_statistic(self, data: BenchmarkData) -> None:
        results = get_face_mesh().process(data.image)
        landmarks = landmarks_from_results(results)

        if not landmarks:
            data.add_property(AngleVarianceProperties.PITCH, None)
            data.add_property(AngleVarianceProperties.YAW, None)
            data.add_property(AngleVarianceProperties.ROLL, None)
            return

        landmarks = landmarks[0]

        geometry = FaceGeometry.estimate_face_geometry(
            self.environment, landmarks, data.image.shape[1], data.image.shape[0]
        )

        transform_matrix = geometry.pose_transform_matrix

        # We need to flip some values to make the angle calculation work
        # See: https://github.com/google/mediapipe/issues/1379#issuecomment-752534379
        transform_matrix[1:3, :] *= -1

        # Get the rotation vector from the rotation matrix
        rotation_vector, _ = cv2.Rodrigues(transform_matrix[:3, :3])

        # Convert to degrees
        angle = np.degrees(rotation_vector)

        data.add_property(AngleVarianceProperties.PITCH, angle[0].item())
        data.add_property(AngleVarianceProperties.YAW, angle[1].item())
        data.add_property(AngleVarianceProperties.ROLL, angle[2].item())

    def collate_statistics(self, data_bin: DataBin[BenchmarkData]) -> Any:
        return {
            "pitch": self._collate_list_statistics(
                data_bin, AngleVarianceProperties.PITCH, ignore_none=True
            ),
            "yaw": self._collate_list_statistics(
                data_bin, AngleVarianceProperties.YAW, ignore_none=True
            ),
            "roll": self._collate_list_statistics(
                data_bin, AngleVarianceProperties.ROLL, ignore_none=True
            ),
        }

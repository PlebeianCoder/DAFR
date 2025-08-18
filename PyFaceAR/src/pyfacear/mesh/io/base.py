from __future__ import annotations

from abc import ABCMeta
from abc import abstractmethod
from os import PathLike
from pathlib import Path
from typing import Union

from pyfacear.mesh.base import Mesh


class MeshIO(metaclass=ABCMeta):
    @classmethod
    def load(cls, path: Union[str, PathLike]) -> Mesh:
        return cls.load_raw(Path(path).read_text())

    @staticmethod
    @abstractmethod
    def load_raw(data: str) -> Mesh:
        pass

    @classmethod
    def export(cls, mesh: Mesh, path: Union[str, PathLike]):
        Path(path).write_text(cls.export_raw(mesh))

    @staticmethod
    @abstractmethod
    def export_raw(mesh: Mesh) -> str:
        pass

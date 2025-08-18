__all__ = [
    "AccessoriesSupplier",
    "from_arguments",
    "AccessoryGenerator",
    "pairwise_impersonation",
    "random_pairwise_impersonation",
    "dodge_each",
    "random_dodge_each",
]

from abc import ABCMeta
from abc import abstractmethod
from random import choices
from typing import Optional, List

from advfaceutil.benchmark.args import AccessoryArguments, BenchmarkArguments
from advfaceutil.benchmark.data import Accessory
from advfaceutil.benchmark.gpu import GPUDeviceManager
from advfaceutil.datasets import FaceDatasetSize
from advfaceutil.datasets import FaceDatasets


class AccessoriesSupplier(metaclass=ABCMeta):
    @abstractmethod
    def supply(self, gpu_manager: GPUDeviceManager) -> List[Accessory]:
        pass


class SimpleAccessoriesSupplier(AccessoriesSupplier):
    def __init__(self, args: List[Accessory]):
        self.args = args

    def supply(self, gpu_manager: GPUDeviceManager) -> List[Accessory]:
        return self.args


def from_arguments(
    accessory_arguments: AccessoryArguments, benchmark_arguments: BenchmarkArguments
) -> List[AccessoriesSupplier]:
    # Load the base class and target class indices
    accessory_arguments.base_class_index = benchmark_arguments.size.class_names.index(
        accessory_arguments.base_class
    )
    if accessory_arguments.target_class is not None:
        accessory_arguments.target_class_index = (
            benchmark_arguments.size.class_names.index(accessory_arguments.target_class)
        )

    return [SimpleAccessoriesSupplier([Accessory.from_arguments(accessory_arguments)])]


class AccessoryGenerator(metaclass=ABCMeta):
    @abstractmethod
    def generate(
        self,
        gpu_manager: GPUDeviceManager,
        dataset: FaceDatasets,
        size: FaceDatasetSize,
        base_class: int,
        target_class: Optional[int],
        base_class_name: str,
        target_class_name: Optional[str],
        universal_class_indices: Optional[List[int]] = None,
    ) -> List[Accessory]:
        pass


class AccessoriesSupplierFromGenerator(AccessoriesSupplier):
    def __init__(
        self,
        generator: AccessoryGenerator,
        dataset: FaceDatasets,
        size: FaceDatasetSize,
        base_class: int,
        target_class: Optional[int],
        base_class_name: str,
        target_class_name: Optional[str],
        universal_class_indices: Optional[List[int]] = None,
    ):
        self.generator = generator
        self.dataset = dataset
        self.size = size
        self.base_class = base_class
        self.target_class = target_class
        self.base_class_name = base_class_name
        self.target_class_name = target_class_name
        self.universal_class_indices = universal_class_indices

    def supply(self, gpu_manager: GPUDeviceManager) -> List[Accessory]:
        return self.generator.generate(
            gpu_manager,
            self.dataset,
            self.size,
            self.base_class,
            self.target_class,
            self.base_class_name,
            self.target_class_name,
            self.universal_class_indices,
        )


def pairwise_impersonation(
    dataset: FaceDatasets,
    size: FaceDatasetSize,
    accessory_generator: AccessoryGenerator,
    researchers_only: bool = False,
) -> List[AccessoriesSupplier]:
    suppliers = []

    if researchers_only:
        classes = list(range(len(size.dataset_names), size.classes))
    else:
        classes = list(range(size.classes))

    for base_class in classes:
        for target_class in range(size.classes):
            # Don't test impersonation from a class to itself and don't target a researcher class if researchers only
            if base_class == target_class or (
                researchers_only and target_class in classes
            ):
                continue

            # Get the class names for the class indices
            base_class_name = size.class_names[base_class]
            target_class_name = size.class_names[target_class]

            suppliers.append(
                AccessoriesSupplierFromGenerator(
                    accessory_generator,
                    dataset,
                    size,
                    base_class,
                    target_class,
                    base_class_name,
                    target_class_name,
                )
            )

    return suppliers


def random_pairwise_impersonation(
    dataset: FaceDatasets,
    size: FaceDatasetSize,
    accessory_generator: AccessoryGenerator,
    class_count: int,
    class_names: Optional[List[str]] = None,
    researchers_only: bool = False,
    base_class_names=None,
) -> List[AccessoriesSupplier]:
    if class_names is None:
        # Choose a random number of classes to impersonate
        if researchers_only:
            universal_class_indices = choices(
                list(range(len(size.dataset_names))), k=class_count
            )
        else:
            universal_class_indices = choices(list(range(size.classes)), k=class_count)
    else:
        universal_class_indices = [size.class_names.index(name) for name in class_names]
        universal_class_indices = universal_class_indices[:class_count]

    if researchers_only:
        classes = list(range(len(size.dataset_names), size.classes))
    elif base_class_names is not None:
        classes = [size.class_names.index(name) for name in base_class_names]
    else:
        classes = list(range(size.classes))

    suppliers = []

    for base_class in classes:
        for target_class in universal_class_indices:
            # Don't test impersonation from a class to itself and don't target a researcher class if researchers only
            if base_class == target_class or (
                researchers_only and target_class in classes
            ):
                continue

            # Get the class names for the class indices
            base_class_name = size.class_names[base_class]
            target_class_name = size.class_names[target_class]

            suppliers.append(
                AccessoriesSupplierFromGenerator(
                    accessory_generator,
                    dataset,
                    size,
                    base_class,
                    target_class,
                    base_class_name,
                    target_class_name,
                    universal_class_indices=universal_class_indices,
                )
            )

    return suppliers


def dodge_each(
    dataset: FaceDatasets,
    size: FaceDatasetSize,
    accessory_generator: AccessoryGenerator,
    researchers_only: bool = False,
) -> List[AccessoriesSupplier]:
    suppliers = []

    if researchers_only:
        classes = range(len(size.dataset_names), size.classes)
    else:
        classes = range(size.classes)

    for base_class in classes:
        base_class_name = size.class_names[base_class]

        suppliers.append(
            AccessoriesSupplierFromGenerator(
                accessory_generator,
                dataset,
                size,
                base_class,
                None,
                base_class_name,
                None,
            )
        )

    return suppliers


# def random_dodge_each(
#     dataset: FaceDatasets,
#     size: FaceDatasetSize,
#     accessory_generator: AccessoryGenerator,
#     class_count: int,
#     class_names: Optional[List[str]] = None,
#     researchers_only: bool = False,
# ) -> List[AccessoriesSupplier]:
#     if class_names is None:
#         # Choose a random number of classes to impersonate
#         universal_class_indices = choices(list(range(size.classes)), k=class_count)
#     else:
#         universal_class_indices = [size.class_names.index(name) for name in class_names]
#         universal_class_indices = universal_class_indices[:class_count]

#     if researchers_only:
#         classes = list(range(len(size.dataset_names), size.classes))
#     else:
#         classes = list(range(size.classes))

#     suppliers = []

#     for base_class in classes:
#         # Get the class names for the class indices
#         base_class_name = size.class_names[base_class]

#         suppliers.append(
#             AccessoriesSupplierFromGenerator(
#                 accessory_generator,
#                 dataset,
#                 size,
#                 base_class,
#                 None,
#                 base_class_name,
#                 None,
#                 universal_class_indices=universal_class_indices,
#             )
#         )

#     return suppliers


def random_dodge_each(
    dataset: FaceDatasets,
    size: FaceDatasetSize,
    accessory_generator: AccessoryGenerator,
    class_count: int,
    class_names: Optional[List[str]] = None,
    researchers_only: bool = False,
) -> List[AccessoriesSupplier]:
    if class_names is None:
        # Choose a random number of classes to impersonate
        universal_class_indices = choices(list(range(size.classes)), k=class_count)
    else:
        universal_class_indices = [size.class_names.index(name) for name in class_names]
        universal_class_indices = universal_class_indices[:class_count]

    if researchers_only:
        classes = list(range(len(size.dataset_names), size.classes))
    else:
        classes = universal_class_indices

    suppliers = []

    for base_class in classes:
        # Get the class names for the class indices
        base_class_name = size.class_names[base_class]

        suppliers.append(
            AccessoriesSupplierFromGenerator(
                accessory_generator,
                dataset,
                size,
                base_class,
                None,
                base_class_name,
                None,
                universal_class_indices=universal_class_indices,
            )
        )

    return suppliers

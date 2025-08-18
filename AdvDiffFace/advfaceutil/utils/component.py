__all__ = [
    "GlobalArguments",
    "Component",
    "ComponentArguments",
    "ComponentEnum",
    "run",
    "run_component",
]

from abc import ABCMeta, abstractmethod
from typing import TypeVar, Type, List
from typing import Optional
from argparse import ArgumentParser, Namespace
from enum import Enum


class GlobalArguments(metaclass=ABCMeta):
    @staticmethod
    @abstractmethod
    def parse_args(args: Namespace) -> None:
        pass

    @staticmethod
    @abstractmethod
    def add_args(parser: ArgumentParser) -> None:
        pass


class ComponentArguments(metaclass=ABCMeta):
    @staticmethod
    @abstractmethod
    def parse_args(args: Namespace) -> "ComponentArguments":
        pass

    @staticmethod
    @abstractmethod
    def add_args(parser: ArgumentParser) -> None:
        pass


ARGS = TypeVar("ARGS", bound=ComponentArguments)


class Component(metaclass=ABCMeta):
    @staticmethod
    @abstractmethod
    def run(args: ARGS) -> None:
        pass


class ComponentEnum(Enum):
    def __init__(
        self, argument_type: Type[ComponentArguments], component_type: Type[Component]
    ) -> None:
        self.argument_type = argument_type
        self.component_type = component_type

    def run(self, args: Namespace) -> None:
        self.component_type.run(self.argument_type.parse_args(args))


def run(
    components: Type[ComponentEnum],
    global_arguments: Optional[List[Type[GlobalArguments]]] = None,
) -> None:
    parser = ArgumentParser(description="Run a component")

    if global_arguments:
        for global_argument in global_arguments:
            global_argument.add_args(parser)

    sub_parsers = parser.add_subparsers(help="Component help")
    for component in components:
        name = component.name.lower()
        sub_parser = sub_parsers.add_parser(name, help=f"{name} help")
        component.argument_type.add_args(sub_parser)
        # Set the function which will be called when these args are detected
        sub_parser.set_defaults(func=component.run)

    parsed_args = parser.parse_args()

    if "func" not in parsed_args:
        print("You must specify an attack program to run!")
        parser.print_help()
        return

    if global_arguments:
        for global_argument in global_arguments:
            global_argument.parse_args(parsed_args)

    # Run the attack program
    parsed_args.func(parsed_args)


def run_component(
    component: Type[Component],
    arguments: Type[ComponentArguments],
    global_arguments: Optional[List[Type[GlobalArguments]]] = None,
) -> None:
    parser = ArgumentParser(description="Run a component")

    if global_arguments:
        for global_argument in global_arguments:
            global_argument.add_args(parser)

    arguments.add_args(parser)

    parsed_args = parser.parse_args()

    if global_arguments:
        for global_argument in global_arguments:
            global_argument.parse_args(parsed_args)

    component.run(arguments.parse_args(parsed_args))

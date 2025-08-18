__all__ = ["LoadResearchers", "SetLogLevel", "DEFAULT_GLOBAL_ARGUMENTS"]

import sys
from argparse import ArgumentParser
from argparse import Namespace
from logging import CRITICAL
from logging import DEBUG
from logging import ERROR
from logging import INFO
from logging import WARNING
from logging import getLevelName
from logging import getLogger, StreamHandler, Formatter

from advfaceutil.utils.component import GlobalArguments


class LoadResearchers(GlobalArguments):
    @staticmethod
    def parse_args(args: Namespace) -> None:
        from advfaceutil.datasets import set_researchers

        if args.researchers:
            researchers = args.researchers
            researchers = researchers.split(",")
            researchers = [researcher.strip() for researcher in researchers]
            set_researchers(researchers)

    @staticmethod
    def add_args(parser: ArgumentParser) -> None:
        parser.add_argument(
            "--researchers",
            type=str,
            help='A comma separated list of researchers names to replace the defaults with. For example: "Name1,Name2"',
        )


class SetLogLevel(GlobalArguments):
    @staticmethod
    def parse_args(args: Namespace) -> None:
        root = getLogger()
        root.setLevel(args.log_level)

        handler = StreamHandler(sys.stdout)
        handler.setLevel(args.log_level)

        formatter = Formatter(
            "%(asctime)s [%(levelname)s] [%(processName)s] [%(name)s] %(message)s",
            "[%d/%m/%Y] [%H:%M:%S]",
        )

        handler.setFormatter(formatter)
        root.addHandler(handler)

    @staticmethod
    def add_args(parser: ArgumentParser) -> None:
        parser.add_argument(
            "--log-level",
            type=str,
            default=getLevelName(INFO),
            choices=list(map(getLevelName, [CRITICAL, ERROR, WARNING, INFO, DEBUG])),
        )
        parser.add_argument(
            "--verbose",
            "-v",
            action="store_const",
            const=getLevelName(DEBUG),
            dest="log_level",
        )


DEFAULT_GLOBAL_ARGUMENTS = [LoadResearchers, SetLogLevel]

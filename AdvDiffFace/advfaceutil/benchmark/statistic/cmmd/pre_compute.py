from argparse import ArgumentParser
from logging import getLogger
from pathlib import Path

import numpy as np

from advfaceutil.benchmark.statistic.cmmd.embedding import ClipEmbeddingModel
from advfaceutil.benchmark.statistic.cmmd.io import (
    compute_embeddings_for_dataset_directories,
)
from advfaceutil.datasets import FaceDatasets
from advfaceutil.utils import SetLogLevel


LOGGER = getLogger("cmmd/pre_compute")


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "datasets_and_directories",
        metavar="N",
        type=str,
        nargs="+",
        help="The datasets and directories to compute embeddings for.",
    )
    parser.add_argument(
        "-bs",
        "--batch-size",
        type=int,
        default=32,
        help="The batch size for the embedding model inference.",
    )
    parser.add_argument(
        "-mcpc",
        "--max-count-per-class",
        type=int,
        default=100,
        help="The maximum number of images in the directories to use for each class.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="embedding.npy",
        help="The path to save the computed embedding to.",
    )

    SetLogLevel.add_args(parser)

    args = parser.parse_args()

    SetLogLevel.parse_args(args)

    LOGGER.info("Loading embedding model")
    embedding_model = ClipEmbeddingModel()

    datasets_and_directories = set(args.datasets_and_directories)

    possible_datasets = [dataset.name for dataset in FaceDatasets]

    datasets = list(
        map(
            lambda d: FaceDatasets[d],
            filter(lambda d: d in possible_datasets, datasets_and_directories),
        )
    )
    directories = datasets_and_directories.difference(datasets)

    directories = [Path(directory) for directory in directories]

    LOGGER.info("Computing embeddings for directories")
    embeddings = compute_embeddings_for_dataset_directories(
        embedding_model,
        directories,
        datasets,
        batch_size=args.batch_size,
        max_count_per_class=args.max_count_per_class,
    )

    np.save(args.output, embeddings)

    LOGGER.info("Saved embeddings to %s", args.output)


if __name__ == "__main__":
    main()

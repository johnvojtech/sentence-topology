import argparse
import enum
import logging
import sys

from sentence_topology.bow import get_embeddings

from .. import utils


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, help="Path to data file.")
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Path where to output embeddings in a tsv format.",
    )
    parser.add_argument(
        "--max_df",
        type=float,
        help="Tokens with higher than given document frequency will get ignored.",
        default=1.0,
    )
    parser.add_argument(
        "--tfidf",
        type=bool,
        action=argparse.BooleanOptionalAction,
        help="Whether to use TF-IDF model. Otherwise plain BOW will be used.",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )

    return args


# TODO: Finish this tomorrow because Magdalena wants to go to sleep.

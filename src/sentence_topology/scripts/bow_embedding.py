import argparse
import logging

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
        help="Tokens with higher than given document frequency will be ignored.",
        default=1.0,
    )
    parser.add_argument(
        "--min_df",
        type=float,
        help="Tokens with lower than given document frequncy will be ignored.",
        default=0.0,
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


def main() -> None:
    args = parse_args()

    logging.info("Loading corpus from %s.", args.input)
    corpus = utils.load_corpus(args.input)
    corpus = list(corpus)

    logging.info("Generating embeddings.")
    embeddings = get_embeddings(
        corpus,
        vectorizer_kwargs={"min_df": args.min_df, "max_df": args.max_df},
        tfidf=args.tfidf,
    )

    logging.info("Saving embeddings into %s.", args.output)
    utils.save_embeddings(embeddings, args.output)


if __name__ == "__main__":
    main()

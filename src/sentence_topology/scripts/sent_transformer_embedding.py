import argparse
import logging

from .. import utils
from .. import sentence_transformers


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, help="Path to data file.")
    parser.add_argument(
        "--model",
        type=str,
        default="paraphrase-multilingual-MiniLM-L12-v2",
        help="Pretrained sentence-transformers model.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Path where to output embeddings in a tsv format.",
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

    all_embeds = sentence_transformers.get_embeddings(corpus, args.model, verbose=True)

    logging.info("Saving embeddings into %s.", args.output)
    utils.save_embeddings(all_embeds, args.output)


if __name__ == "__main__":
    main()

import argparse
import enum
import logging
import sys

from sentence_topology.sentence_transformers.embeddings import (
    get_embeddings, get_embeddings_trans_prediction)

from .. import utils


class TrainObjectives(enum.Enum):
    TransformationPrediction = "transformation-prediction"


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
        help=(
            "Path where to output embeddings in a tsv format. Given path can have named"
            " placeholders. Available placeholders are:\n\t-'model' - model"
            " name\n\t-'split_ind' - underscore and split index for supervised"
            " embeddings."
        ),
        default="./embeddings/{model}{split_ind}.tsv",
    )
    parser.add_argument(
        "--train_objective",
        type=str,
        default=None,
        help=(
            "Objective to pretrain on. Available objectives:"
            f" {','.join([o.value for o in TrainObjectives])}"
        ),
    )
    parser.add_argument(
        "--logdir",
        type=str,
        default=None,
        help="Directory where to output tensorboard logs for supervised embeddings.",
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

    if args.train_objective is None:
        all_embeds = get_embeddings(corpus, args.model, verbose=True)

        output_file = args.output.format(model=args.model, split_ind="")
        logging.info("Saving embeddings into %s.", output_file)
        utils.save_embeddings(all_embeds, output_file)
        sys.exit(0)

    if args.train_objective == TrainObjectives.TransformationPrediction:
        embeddings_by_split = get_embeddings_trans_prediction(
            corpus,
            args.model,
            verbose=True,
            log_dir=args.logdir,
            epochs=4,
        )
        for split_ind, embeds in enumerate(embeddings_by_split):
            output_file = args.output.format(
                model=args.model, split_ind=f"_{split_ind}"
            )
            logging.info("Saving embeddings into %s.", output_file)
            utils.save_embeddings(embeds, output_file)


if __name__ == "__main__":
    main()

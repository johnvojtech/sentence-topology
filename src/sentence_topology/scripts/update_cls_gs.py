import argparse
import logging
import os
from typing import Callable, Iterable

import pandas as pd
from tqdm.auto import tqdm

from sentence_topology.classification.analysis import (
    ClassifierAnalysisResults,
    analyze_classifier,
)
from sentence_topology.classification.grid_search import (
    DEFAULT_GRID_SEARCHED_CLASSIFIERS,
    GridSearchClassifier,
    grid_search_classifiers_params,
)
from sentence_topology.data_types import CostraEmbedding
from sentence_topology.utils.io import load_embedding
from sentence_topology.utils.transform import CONTEXT_MODES, contextualize_embeddings

logger = logging.getLogger(__name__)


def update_grid_search_scores_and_params(
    scores: pd.DataFrame,
    params: pd.DataFrame,
    embeddings: list[tuple[str, list[CostraEmbedding]]],
    *,
    scoring: str,
) -> None:
    for name, embedding in tqdm(embeddings):
        evals = grid_search_classifiers_params(
            embedding,
            DEFAULT_GRID_SEARCHED_CLASSIFIERS,
            scoring=scoring,
        )
        scores.loc[name] = [eval.best_score_ for eval in evals]
        params.loc[name] = [eval.best_params_ for eval in evals]


def update_analysis_results(
    *,
    scores: pd.DataFrame,
    params: pd.DataFrame,
    embeddings: list[tuple[str, list[CostraEmbedding]]],
    analysis_results_dir: str,
    analysis_figs_dir: str,
    gs_cls_map: dict[str, GridSearchClassifier],
    scoring: str,
) -> None:
    for embed_name, embedding in tqdm(embeddings):
        embed_basename = embed_name[: embed_name.rfind(".")]
        analysis_save_path = os.path.join(analysis_results_dir, f"{embed_basename}.pkl")

        best_classifier_name = scores.loc[embed_name].idxmax()

        cls_params = params.loc[embed_name, best_classifier_name]
        classifier = gs_cls_map[best_classifier_name].estimator
        classifier.set_params(**cls_params)

        analysis = analyze_classifier(embedding, classifier, scoring=scoring)
        analysis.classifier_params = cls_params

        analysis.save(analysis_save_path)

        fig_save_path = os.path.join(analysis_figs_dir, f"{embed_basename}.png")
        fig = analysis.visualize()
        fig.savefig(
            fig_save_path,
            bbox_inches="tight",
        )


def get_embedding_entries(embeddings_dir: str) -> Iterable[os.DirEntry]:
    for entry in os.scandir(embeddings_dir):
        if entry.is_file() and entry.name.endswith(".tsv"):
            yield entry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Updates grid search and analysis classification results for all new"
            " embeddings."
        )
    )
    parser.add_argument(
        "--context_mode",
        type=str,
        help=(
            "Mode of contextualization to apply on the embeddings before feeding them"
            f" to classifiers. Available options are: {','.join(CONTEXT_MODES.keys())}."
        ),
        default=None,
    )
    parser.add_argument(
        "--update_grid_search",
        type=bool,
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Update grid search scores and parameters.",
    )
    parser.add_argument(
        "--update_analysis",
        type=bool,
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Update analysis of best classifiers.",
    )
    parser.add_argument(
        "--embeddings_dir",
        type=str,
        help="Path to directory containing all embeddings.",
        default="./embeddings",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        help=(
            "Path to directory for storing results. Default is"
            " './results/cls_gs/`context_mode`'"
        ),
        default=None,
    )
    parser.add_argument(
        "--analysis_results_dir_name",
        type=str,
        help=(
            "Path to directory where to save analysis of best classifiers results under"
            " `results_dir`."
        ),
        default="best_classifier_analysis",
    )
    parser.add_argument(
        "--analysis_figs_dir_name",
        type=str,
        help=(
            "Path to directory where to save analysis figures of best classifiers under"
            " `results_dir`."
        ),
        default=os.path.join(
            "best_classifier_analysis",
            "figs",
        ),
    )
    parser.add_argument(
        "--gs_scores",
        type=str,
        help=(
            "Path to pickled DataFrame containing embedding-classifier table of highest"
            " scores."
        ),
        required=True,
    )
    parser.add_argument(
        "--gs_params",
        type=str,
        help=(
            "Path to pickled DataFrame containing embedding-classifier table of best"
            " params."
        ),
        required=True,
    )
    parser.add_argument(
        "--scoring",
        type=str,
        help="Scoring method to identify better classifiers.",
        default="accuracy",
    )

    args = parser.parse_args()

    if args.results_dir is None:
        args.results_dir = os.path.join(
            ".",
            "results",
            "cls_gs",
            args.context_mode if args.context_mode is not None else "no_context",
        )
    os.makedirs(args.results_dir, exist_ok=True)

    args.analysis_results_dir = os.path.join(
        args.results_dir,
        args.analysis_results_dir_name,
    )
    os.makedirs(args.analysis_results_dir, exist_ok=True)

    args.analysis_figs_dir = os.path.join(
        args.results_dir,
        args.analysis_figs_dir_name,
    )
    os.makedirs(args.analysis_figs_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )

    return args


def main() -> None:
    args = parse_args()

    if not os.path.exists(args.gs_scores):
        logger.info("Creating new scores at %s.", args.gs_scores)

    scores = (
        pd.read_pickle(args.gs_scores)
        if os.path.exists(args.gs_scores)
        else pd.DataFrame(columns=list(DEFAULT_GRID_SEARCHED_CLASSIFIERS.keys()))
    )

    if not os.path.exists(args.gs_params):
        logger.info("Creating new params at %s.", args.gs_params)

    params = (
        pd.read_pickle(args.gs_params)
        if os.path.exists(args.gs_params)
        else pd.DataFrame(columns=list(DEFAULT_GRID_SEARCHED_CLASSIFIERS.keys()))
    )

    def fetch_embedding(name: str) -> tuple[list[CostraEmbedding], bool]:
        embedding = list(load_embedding(os.path.join(args.embeddings_dir, name)))
        valid = True
        if args.context_mode is not None:
            embedding, skipped_count = contextualize_embeddings(
                embedding, mode=args.context_mode
            )
            valid = False
            if skipped_count > 0:
                logger.warn(
                    "Unable to contextualize %d embeddings from %s. Skipping",
                    skipped_count,
                    name,
                )

        return embedding, valid

    embeds_to_update = []
    for entry in get_embedding_entries(args.embeddings_dir):
        if entry.name not in scores.index or entry.name not in params.index:
            embedding, valid = fetch_embedding(entry.name)
            if valid and (
                entry.name not in scores.index or entry.name not in params.index
            ):
                embeds_to_update.append((entry.name, embedding))

    logger.info("Updating %d embeddings.", len(embeds_to_update))
    if args.update_grid_search:
        logger.info("Updating grid search scores and params.")

        update_grid_search_scores_and_params(
            scores, params, embeds_to_update, scoring=args.scoring
        )
        scores.to_pickle(args.gs_scores)
        params.to_pickle(args.gs_params)

    scores = pd.read_pickle(args.gs_scores)
    params = pd.read_pickle(args.gs_params)

    if args.update_analysis:
        update_analysis_results(
            scores=scores,
            params=params,
            embeddings=embeds_to_update,
            analysis_results_dir=args.analysis_results_dir,
            analysis_figs_dir=args.analysis_figs_dir,
            scoring=args.scoring,
            gs_cls_map=DEFAULT_GRID_SEARCHED_CLASSIFIERS,
        )


if __name__ == "__main__":
    main()

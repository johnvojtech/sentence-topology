import argparse
import logging
import os

import pandas as pd
from tqdm.auto import tqdm

from sentence_topology.classification.analysis import (
    ClassifierAnalysisResults, analyze_classifier)
from sentence_topology.classification.grid_search import (
    DEFAULT_GRID_SEARCHED_CLASSIFIERS, grid_search_classifiers_params)
from sentence_topology.utils.io import load_embedding


def update_grid_search_scores_and_params(
    scores: pd.DataFrame, params: pd.DataFrame, *, embeddings_dir: str
) -> None:
    def should_update(entry: os.DirEntry) -> bool:
        return (
            entry.is_file()
            and entry.name.endswith(".tsv")
            and (entry.name not in scores.index or entry.name not in params.index)
        )

    for entry in tqdm(os.scandir(embeddings_dir)):
        if should_update(entry):
            embeddings = list(load_embedding(entry.path))
            evals = grid_search_classifiers_params(
                embeddings, DEFAULT_GRID_SEARCHED_CLASSIFIERS
            )
            scores.loc[entry.name] = [eval.best_score_ for eval in evals]
            params.loc[entry.name] = [eval.best_params_ for eval in evals]


def update_analysis_results(
    scores: pd.DataFrame,
    params: pd.DataFrame,
    *,
    embeddings_dir: str,
    analysis_results_dir: str,
    analysis_fig_dir: str,
) -> None:
    cls_type_by_name = {}
    for classifier_config in DEFAULT_GRID_SEARCHED_CLASSIFIERS:
        cls_type_by_name[classifier_config.type_.__name__] = classifier_config.type_

    for embed_name, embed_scores in tqdm(scores.iterrows(), total=scores.shape[0]):
        embed_name = str(embed_name)
        embed_file_name = embed_name[: embed_name.rfind(".")]
        analysis_save_path = os.path.join(
            analysis_results_dir, f"{embed_file_name}.pkl"
        )

        if not os.path.exists(analysis_save_path):
            best_classifier_name = embed_scores.idxmax()

            cls_params = params.loc[embed_name, best_classifier_name]
            classifier = cls_type_by_name[best_classifier_name](**cls_params)
            embeddings = load_embedding(os.path.join(embeddings_dir, embed_name))

            analysis = analyze_classifier(list(embeddings), classifier)
            analysis.classifier_params = params.loc[embed_name, best_classifier_name]

            analysis.save(analysis_save_path)

        analysis = ClassifierAnalysisResults.load(analysis_save_path)
        fig_save_path = os.path.join(analysis_fig_dir, f"{embed_file_name}.png")
        if not os.path.exists(fig_save_path):
            fig = analysis.visualize()
            fig.savefig(
                fig_save_path,
                bbox_inches="tight",
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Updates grid search and analysis classification results for all new"
            " embeddings."
        )
    )
    parser.add_argument(
        "--embeddings_dir",
        type=str,
        help="Path to directory containing all embeddings.",
        defualt="./embeddings",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        help="Path to directory for storing results.",
        default="./results",
    )
    parser.add_argument(
        "--analysis_results_dir",
        type=str,
        help=(
            "Path to directory where to save analysis results of best classifiers. If"
            " not given '`results_dir`/best_classifier_analysis/' is used."
        ),
        default=None,
    )
    parser.add_argument(
        "--analysis_figs_dir",
        type=str,
        help=(
            "Path to directory where to save analysis figures of best classifiers. If"
            " not given '`results_dir`/best_classifier_analysis/figs/' is used."
        ),
        default=None,
    )
    parser.add_argument(
        "--gs_scores",
        type=str,
        help=(
            "Path to pickled DataFrame containing embedding-classifier table of highest"
            " scores."
        ),
    )
    parser.add_argument(
        "--gs_params",
        type=str,
        help=(
            "Path to pickled DataFrame containing embedding-classifier table of best"
            " params."
        ),
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

    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)

    if args.analysis_results_dir is None:
        args.analysis_results_dir = os.path.join(
            args.results_dir, "best_classifier_analysis"
        )

    os.makedirs(args.analysis_results_dir, exist_ok=True)
    if args.analysis_fig_dir is None:
        args.analysis_fig_dir = os.path.join(args.analysis_fig_dir, "figs")
    os.makedirs(args.analysis_fig_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )

    return args


def main() -> None:
    args = parse_args()

    scores = (
        pd.read_pickle(args.gs_scores)
        if os.path.exists(args.gs_scores)
        else pd.DataFrame()
    )
    params = (
        pd.read_pickle(args.gs_params)
        if os.path.exists(args.gs_params)
        else pd.DataFrame()
    )

    if args.update_grid_search:
        update_grid_search_scores_and_params(
            scores, params, embeddings_dir=args.embeddings_dir
        )
        scores.to_pickle(args.gs_scores)
        params.to_pickle(args.gs_params)

    scores = pd.read_pickle(args.gs_scores)
    params = pd.read_pickle(args.params)

    if args.update_analysis:
        update_analysis_results(
            scores,
            params,
            embeddings_dir=args.embeddings_dir,
            analysis_results_dir=args.analysis_results_dir,
            analysis_fig_dir=args.analysis_figs_dir,
        )


if __name__ == "__main__":
    main()

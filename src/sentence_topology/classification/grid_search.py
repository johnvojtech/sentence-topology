from dataclasses import dataclass
from typing import Any

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network._multilayer_perceptron import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from tqdm.auto import tqdm

from sentence_topology.classification.analysis import \
    create_embedding_transformation_prediction_data
from sentence_topology.data_types import CostraEmbedding
from sentence_topology.utils import load_all_embeddings


@dataclass
class GridSearchClassifier:
    classifier_type: type
    params: dict[str, list[Any]]


@dataclass
class EvaluatedGridSearchClassifier:
    classifier_type: type
    best_params_: dict[str, Any]
    best_score_: float


def grid_search_classifiers_params(
    embeddings: list[CostraEmbedding],
    clsfiers: list[GridSearchClassifier],
    *,
    with_bar: bool = True,
    verbose: int = 0,
    scoring: str = "accuracy",
    paralel: bool = True,
) -> list[EvaluatedGridSearchClassifier]:
    data = create_embedding_transformation_prediction_data(embeddings)

    evaluated = []
    with tqdm(
        clsfiers, desc="Classifiers evaluated", disable=not with_bar
    ) as progress_bar:
        for classifier in progress_bar:
            progress_bar.set_postfix(classifier=classifier.classifier_type.__name__)
            gs = GridSearchCV(
                classifier.classifier_type(),
                classifier.params,
                scoring=scoring,
                cv=StratifiedGroupKFold(),
                verbose=verbose,
            )

            def _fit() -> None:
                gs.fit(data.features, data.labels, groups=data.groups)

            if paralel:
                with joblib.parallel_backend(backend="loky"):
                    _fit()
            else:
                _fit()

            evaluated.append(
                EvaluatedGridSearchClassifier(
                    classifier_type=classifier.classifier_type,
                    best_params_=gs.best_params_,
                    best_score_=gs.best_score_,
                )
            )

    return evaluated


DEFAULT_GRID_SEARCHED_CLASSIFIERS = [
    GridSearchClassifier(
        classifier_type=DecisionTreeClassifier,
        params={
            "max_depth": [6, 18, 32],
            "min_samples_split": [2, 5],
            "max_leaf_nodes": [70, 50, 20],
        },
    ),
    GridSearchClassifier(
        classifier_type=MLPClassifier,
        params={
            "hidden_layer_sizes": [(25,), (50,), (50, 25), (25, 5)],
            "activation": ["relu", "logistic"],
            "max_iter": [1000],
        },
    ),
    GridSearchClassifier(
        classifier_type=RandomForestClassifier,
        params={
            "n_estimators": [50, 100, 200],
            "max_depth": [2, 5, 25, None],
            "min_samples_split": [5, 10, 20],
        },
    ),
    GridSearchClassifier(
        classifier_type=SVC,
        params={
            "kernel": ["rbf", "linear"],
            "gamma": ["auto", "scale"],
        },
    ),
    GridSearchClassifier(
        classifier_type=KNeighborsClassifier,
        params={
            "n_neighbors": [3, 5, 10],
            "weights": ["uniform", "distance"],
        },
    ),
]


def grid_search_classifiers_params_for_all_embeddings(
    embeddings_dir: str,
    classifiers: list[GridSearchClassifier] = DEFAULT_GRID_SEARCHED_CLASSIFIERS,
    *,
    with_bar: bool = True,
    paralel: bool = True,
    scoring: str = "accuracy",
) -> dict[str, list[EvaluatedGridSearchClassifier]]:
    evaluated = {}
    with tqdm(
        list(load_all_embeddings(embeddings_dir)),
        desc="Embedding",
        disable=not with_bar,
    ) as bar:
        for entry, embedding in bar:
            bar.set_postfix(embedding=entry.name)
            evaluated[entry.name] = grid_search_classifiers_params(
                list(embedding),
                classifiers,
                verbose=0,
                with_bar=with_bar,
                scoring=scoring,
                paralel=paralel,
            )

    return evaluated

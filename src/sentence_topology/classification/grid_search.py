from dataclasses import dataclass
from typing import Any

import joblib
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network._multilayer_perceptron import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

from sentence_topology.classification.analysis import (
    create_embedding_transformation_prediction_data,
)
from sentence_topology.data_types import CostraEmbedding


@dataclass
class GridSearchClassifier:
    estimator: BaseEstimator
    params: dict[str, list[Any]]


@dataclass
class EvaluatedGridSearchClassifier:
    estimator: BaseEstimator
    best_params_: dict[str, Any]
    best_score_: float


def grid_search_classifiers_params(
    embeddings: list[CostraEmbedding],
    clsfiers: dict[str, GridSearchClassifier],
    *,
    with_bar: bool = True,
    verbose: int = 0,
    scoring: str = "accuracy",
    paralel: bool = True,
) -> list[EvaluatedGridSearchClassifier]:
    data = create_embedding_transformation_prediction_data(embeddings)

    evaluated = []
    with tqdm(
        clsfiers.items(), desc="Classifiers evaluated", disable=not with_bar
    ) as progress_bar:
        for cls_name, classifier in progress_bar:
            progress_bar.set_postfix(classifier=cls_name)
            gs = GridSearchCV(
                classifier.estimator,
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
                    estimator=classifier.estimator,
                    best_params_=gs.best_params_,
                    best_score_=gs.best_score_,
                )
            )

    return evaluated


DEFAULT_GRID_SEARCHED_CLASSIFIERS = {
    "DecisionTreeClassifier": GridSearchClassifier(
        estimator=DecisionTreeClassifier(),
        params={
            "max_depth": [6, 18, 32],
            "min_samples_split": [2, 5],
            "max_leaf_nodes": [70, 50, 20],
        },
    ),
    "MLPClassifier": GridSearchClassifier(
        estimator=MLPClassifier(),
        params={
            "hidden_layer_sizes": [(25,), (50,), (50, 25), (25, 5)],
            "activation": ["relu", "logistic"],
            "max_iter": [1000],
        },
    ),
    "RandomForestClassifier": GridSearchClassifier(
        estimator=RandomForestClassifier(),
        params={
            "n_estimators": [50, 100, 200],
            "max_depth": [2, 5, 25, None],
            "min_samples_split": [5, 10, 20],
        },
    ),
    "SVC": GridSearchClassifier(
        estimator=Pipeline([("scaler", StandardScaler()), ("svc", SVC())]),
        params={
            "svc__kernel": ["rbf", "linear"],
            "svc__gamma": ["auto", "scale"],
        },
    ),
    "KNeighborsClassifier": GridSearchClassifier(
        estimator=Pipeline(
            [("scaler", StandardScaler()), ("knn", KNeighborsClassifier())]
        ),
        params={
            "knn__n_neighbors": [3, 5, 10],
            "knn__weights": ["uniform", "distance"],
        },
    ),
}

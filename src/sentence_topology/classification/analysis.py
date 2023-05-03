from __future__ import annotations

import pickle
from dataclasses import dataclass
from typing import Any, cast

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, get_scorer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from sentence_topology.data_types import CostraEmbedding
from sentence_topology.visualization.predictions import (
    draw_confusion_matrix,
    draw_distributions,
)


@dataclass
class EmbeddingTransformationPredictionData:
    features: np.ndarray
    labels: np.ndarray
    label_encoder: LabelEncoder
    groups: np.ndarray


def create_embedding_transformation_prediction_data(
    embeddings: list[CostraEmbedding],
) -> EmbeddingTransformationPredictionData:
    features = []
    labels = []
    groups = []

    all_transformations = list(set((embed.trans for embed in embeddings)))
    label_encoder = LabelEncoder()
    label_encoder.fit(all_transformations)

    for embed in embeddings:
        features.append(embed.embedding)
        labels.append(label_encoder.transform([embed.trans])[0])
        groups.append(embed.seed_id)

    return EmbeddingTransformationPredictionData(
        features=np.array(features),
        labels=np.array(labels),
        label_encoder=label_encoder,
        groups=np.array(groups),
    )


@dataclass
class ClassifierAnalysisResults:
    classifier_type: type
    classifier_params: dict[str, Any]
    confusion_matrix: pd.DataFrame
    score: float
    score_name: str
    macro_metrics: dict[str, float]
    report: pd.DataFrame

    def save(self, path: str) -> None:
        with open(path, mode="wb") as save_file:
            pickle.dump(self, save_file)

    @classmethod
    def load(cls, path: str) -> ClassifierAnalysisResults:
        with open(path, mode="rb") as save_file:
            return pickle.load(save_file)

    def visualize(self) -> plt.Figure:
        fig = plt.figure(figsize=(12, 14))
        fig.suptitle(
            "\n".join(
                [
                    f"classifier: {self.classifier_type.__name__}",
                    f"{self.score_name}: {self.score:.5f}",
                    f"params: {self.classifier_params}",
                ]
            ),
            x=0.44,
            y=0.95,
        )
        grid = gridspec.GridSpec(
            2,
            2,
            figure=fig,
            width_ratios=[7.5, 1],
            height_ratios=[8, 4],
            hspace=0.6,
        )

        conf_matrix_axis = fig.add_subplot(grid[:-1, :])
        dist_matrix_axis = fig.add_subplot(grid[-1, :-1])

        conf_matrix_axis.set_title("Normalized confusion matrix")
        draw_confusion_matrix(self.confusion_matrix, conf_matrix_axis)
        dist_matrix_axis.set_title("Label distribution")
        draw_distributions(self.confusion_matrix, dist_matrix_axis)

        return fig


def analyze_classifier(
    embeddings: list[CostraEmbedding],
    classifier,
    *,
    test_split_size: float = 0.5,
    scoring: str = "accuracy",
) -> ClassifierAnalysisResults:
    data = create_embedding_transformation_prediction_data(embeddings)
    class_names = cast(list[str], data.label_encoder.classes_)

    feat_train, feat_test, label_train, label_test = train_test_split(
        data.features,
        data.labels,
        test_size=test_split_size,
        stratify=data.labels,
    )
    classifier.fit(feat_train, label_train)

    predictions = classifier.predict(feat_test)
    metrics = cast(
        dict[str, Any],
        classification_report(
            label_test,
            predictions,
            target_names=data.label_encoder.classes_,
            output_dict=True,
        ),
    )

    rows = {}
    for label_name in class_names:
        rows[label_name] = metrics.pop(label_name)

    report = pd.DataFrame(rows)

    conf_matrix = confusion_matrix(label_test, predictions)
    conf_matrix = pd.DataFrame(
        conf_matrix,
        index=class_names,
        columns=class_names,
    )

    score = get_scorer(scoring)(classifier, feat_test, label_test)

    return ClassifierAnalysisResults(
        classifier_type=type(classifier),
        classifier_params=classifier.get_params(deep=False),
        report=report,
        score=score,
        score_name=scoring,
        macro_metrics=cast(dict[str, float], metrics["macro avg"]),
        confusion_matrix=conf_matrix,
    )

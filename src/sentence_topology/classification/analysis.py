from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from sentence_topology.data_types import CostraEmbedding


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
    confusion_matrix: pd.DataFrame
    accuracy: float
    macro_metrics: dict[str, float]
    report: pd.DataFrame

    # TODO: Save method


def analyze_classifier(
    embeddings: list[CostraEmbedding],
    classifier,
    *,
    test_split_size: float = 0.5,
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

    conf_matrix = confusion_matrix(feat_test, predictions)
    conf_matrix = pd.DataFrame(
        conf_matrix,
        index=class_names,
        columns=class_names,
    )

    return ClassifierAnalysisResults(
        report=report,
        accuracy=cast(float, metrics["accuracy"]),
        macro_metrics=cast(dict[str, float], metrics["macro_avg"]),
        confusion_matrix=conf_matrix,
    )

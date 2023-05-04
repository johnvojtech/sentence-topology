from __future__ import annotations

import pickle
from dataclasses import dataclass
from functools import reduce
from typing import Any, cast

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, get_scorer
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.preprocessing import LabelEncoder

from sentence_topology.data_types import CostraEmbedding
from sentence_topology.visualization.predictions import (draw_confusion_matrix,
                                                         draw_distributions)


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
    macro_avg: dict[str, float]
    weighted_avg: dict[str, float]
    accuracy: float
    report: pd.DataFrame

    def save(self, path: str) -> None:
        with open(path, mode="wb") as save_file:
            pickle.dump(self, save_file)

    @classmethod
    def load(cls, path: str) -> ClassifierAnalysisResults:
        with open(path, mode="rb") as save_file:
            return pickle.load(save_file)

    @classmethod
    def from_trained_classifier(
        cls,
        classifier,
        test_inputs,
        test_labels,
        *,
        scoring: str,
        class_names: list[str],
    ) -> ClassifierAnalysisResults:
        predictions = classifier.predict(test_inputs)
        metrics = cast(
            dict[str, Any],
            classification_report(
                test_labels,
                predictions,
                target_names=class_names,
                output_dict=True,
            ),
        )

        rows = {}
        for label_name in class_names:
            rows[label_name] = metrics.pop(label_name)

        report = pd.DataFrame(rows)

        conf_matrix = confusion_matrix(test_labels, predictions)
        conf_matrix = pd.DataFrame(
            conf_matrix,
            index=class_names,
            columns=class_names,
        )

        score = get_scorer(scoring)(classifier, test_inputs, test_labels)

        return cls(
            classifier_type=type(classifier),
            classifier_params=classifier.get_params(deep=False),
            report=report,
            score=score,
            score_name=scoring,
            macro_avg=cast(dict[str, float], metrics["macro avg"]),
            weighted_avg=cast(dict[str, float], metrics["weighted avg"]),
            accuracy=cast(float, metrics["accuracy"]),
            confusion_matrix=conf_matrix,
        )

    @classmethod
    def aggregate(
        cls, *analyses: ClassifierAnalysisResults
    ) -> ClassifierAnalysisResults:
        classifier_type = analyses[0].classifier_type
        classifier_params = analyses[0].classifier_params
        score_name = analyses[0].score_name

        for ana in analyses:
            assert (
                ana.classifier_type == classifier_type
            ), "Analyses came from different classifiers."
            assert (
                ana.classifier_params == classifier_params
            ), "Analyses came from classifiers with different params."
            assert ana.score_name == score_name, "Analyses were scored differently."

        analyses_count = len(analyses)

        confusion_matrix = cls._aggregate_dfs(
            *(ana.confusion_matrix for ana in analyses),
        )
        report = cls._aggregate_dfs(*(ana.report for ana in analyses))

        score = sum(ana.score for ana in analyses) / analyses_count
        accuracy = sum(ana.accuracy for ana in analyses) / analyses_count

        macro_avg = cls._aggregate_dicts(*(ana.macro_avg for ana in analyses))
        weighted_avg = cls._aggregate_dicts(*(ana.weighted_avg for ana in analyses))

        return cls(
            classifier_type=classifier_type,
            classifier_params=classifier_params,
            confusion_matrix=confusion_matrix,
            score_name=score_name,
            score=score,
            report=report,
            macro_avg=macro_avg,
            weighted_avg=weighted_avg,
            accuracy=accuracy,
        )

    @classmethod
    def _aggregate_dfs(cls, *dfs: pd.DataFrame) -> pd.DataFrame:
        def add_df(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
            return a.add(b, fill_value=0)

        df = reduce(
            add_df,
            dfs,
        )
        df /= len(dfs)
        return df

    @classmethod
    def _aggregate_dicts(cls, *dicts: dict[str, float]) -> dict[str, float]:
        required_keys = set(dicts[0].keys())
        agg_dict = {}
        for dic in dicts:
            assert (
                len(set(dic.keys()) - required_keys) == 0
            ), "Dicts have varying keys. Cannot aggregate."
            for key in required_keys:
                agg_dict[key] = dic[key] / len(dicts)

        return agg_dict

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


def _analyze_with_cv(
    *, classifier, data: EmbeddingTransformationPredictionData, scoring: str
) -> ClassifierAnalysisResults:
    cv = StratifiedGroupKFold()

    analyses = []
    for train_idxs, test_idxs in cv.split(
        data.features, data.labels, groups=data.groups
    ):
        train_feats, train_labels = data.features[train_idxs], data.labels[train_idxs]
        test_feats, test_labels = data.features[test_idxs], data.labels[test_idxs]

        classifier.fit(train_feats, train_labels)
        analyses.append(
            ClassifierAnalysisResults.from_trained_classifier(
                classifier,
                test_feats,
                test_labels,
                scoring=scoring,
                class_names=cast(list[str], data.label_encoder.classes_),
            )
        )

    return ClassifierAnalysisResults.aggregate(*analyses)


def _analyze_with_splits(
    *,
    classifier,
    data: EmbeddingTransformationPredictionData,
    test_split_size: float,
    scoring: str,
) -> ClassifierAnalysisResults:
    feat_train, feat_test, label_train, label_test = train_test_split(
        data.features,
        data.labels,
        test_size=test_split_size,
        stratify=data.labels,
    )
    classifier.fit(feat_train, label_train)

    class_names = cast(list[str], data.label_encoder.classes_)
    return ClassifierAnalysisResults.from_trained_classifier(
        classifier, feat_test, label_test, scoring=scoring, class_names=class_names
    )


def analyze_classifier(
    embeddings: list[CostraEmbedding],
    classifier,
    *,
    test_split_size: float = 0.5,
    scoring: str = "accuracy",
    cross_validation: bool = False,
) -> ClassifierAnalysisResults:
    data = create_embedding_transformation_prediction_data(embeddings)

    if not cross_validation:
        return _analyze_with_splits(
            classifier=classifier,
            data=data,
            test_split_size=test_split_size,
            scoring=scoring,
        )

    return _analyze_with_cv(
        classifier=classifier,
        data=data,
        scoring=scoring,
    )

from typing import Any, cast

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, get_scorer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from sentence_topology.classification.analysis import (
    ClassifierAnalysisResults, create_embedding_transformation_prediction_data)
from sentence_topology.utils.io import load_embedding
from sentence_topology.utils.transform import equalize_transformations
from sentence_topology.visualization.predictions import draw_distributions

# %%


embedding = list(
    load_embedding("../embeddings/paraphrase-multilingual-MiniLM-L12-v2.tsv")
)
# %%
embedding = equalize_transformations(embedding)
# %%
data = create_embedding_transformation_prediction_data(embedding)

# %%
scoring = "f1_macro"

# %%
classifier = DecisionTreeClassifier()

feat_train, feat_test, label_train, label_test = train_test_split(
    data.features,
    data.labels,
    test_size=0.5,
    stratify=data.labels,
)
classifier.fit(feat_train, label_train)

class_names = cast(list[str], data.label_encoder.classes_)

predictions = classifier.predict(feat_test)
metrics = cast(
    dict[str, Any],
    classification_report(
        label_test,
        predictions,
        target_names=class_names,
        output_dict=True,
    ),
)
# %%
print(metrics)
# %%

rows = {}
for label_name in class_names:
    rows[label_name] = metrics.pop(label_name)

report = pd.DataFrame(rows)
# %%
report
# %%
metrics
# %%

conf_matrix = confusion_matrix(label_test, predictions)
conf_matrix = pd.DataFrame(
    conf_matrix,
    index=class_names,
    columns=class_names,
)

score = get_scorer(scoring)(classifier, feat_test, label_test)

analysis = ClassifierAnalysisResults(
    classifier_type=type(classifier),
    classifier_params=classifier.get_params(deep=False),
    report=report,
    score=score,
    score_name=scoring,
    confusion_matrix=conf_matrix,
    macro_avg=cast(dict[str, float], metrics["macro avg"]),
    weighted_avg=cast(dict[str, float], metrics["weighted avg"]),
    accuracy=cast(float, metrics["accuracy"]),
)
# %%
preds = analysis.confusion_matrix.sum(axis=0)
true = analysis.confusion_matrix.sum(axis=1)
# %%
print(preds)
# %%
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme()

# %%
sns.barplot(x=preds.index, y=preds, color="tab:blue")
plt.xticks(rotation=90)
# plt.xticklabels(plt.get_xticklabels(), rotation=90)
# %%
sns.barplot(x=true.index, y=true, color="tab:blue")
plt.xticks(rotation=90)
# %%
hists = pd.DataFrame({"Predicted": preds, "True": true, "Label": preds.index})
hists.set_index("Label", inplace=True)
hists
# %%
# sns.catplot(data=hists)
# plt.xticks(rotation=90)
hists.plot.bar()

# %%
fig, axis = plt.subplots(1, 1)
draw_distributions(analysis.confusion_matrix, axis)

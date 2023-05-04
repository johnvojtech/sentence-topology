import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def draw_confusion_matrix(
    conf_matrix: pd.DataFrame,
    axes: plt.Axes,
    *,
    norm: bool = True,
) -> None:
    matrix_norm = conf_matrix.div(conf_matrix.sum(axis=1), axis=0)
    matrix_norm *= 100
    sns.heatmap(
        matrix_norm if norm else conf_matrix,
        annot=True,
        ax=axes,
        fmt=".0f",
    )

    axes.set_ylabel("True")
    axes.set_xlabel("Predicted")


def draw_distributions(conf_matrix: pd.DataFrame, axes: plt.Axes) -> None:
    true = conf_matrix.sum(axis=1)
    pred = conf_matrix.sum(axis=0)
    hists = pd.DataFrame({"Predicted": preds, "True": true, "Label": preds.index})
    hists.set_index("Label", inplace=True)
    hists.plot.bar(ax=axes)
    axes.set_xticks(rotation=90)

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
    types = []
    classes = []

    for type_, counts in zip(["True", "Predicted"], [true, pred]):
        for cls, count in counts.items():
            for _ in range(count):
                types.append(type_)
                classes.append(cls)

    dists = pd.DataFrame({"Distribution": types, "Classes": classes})
    sns.histplot(
        dists,
        x="Classes",
        hue="Distribution",
        element="step",
        stat="percent",
        ax=axes,
    )
    axes.set_xticklabels(axes.get_xticklabels(), rotation=90)

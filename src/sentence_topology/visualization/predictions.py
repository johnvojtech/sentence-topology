import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def confusion_matrix(
    conf_matrix: pd.DataFrame,
    axes: plt.Axes,
    *,
    norm: bool = True,
) -> None:
    axes.set_ylabel("True")
    axes.set_xlabel("Predicted")

    matrix_norm = conf_matrix.div(conf_matrix.sum(axis=1), axis=0)
    matrix_norm *= 100
    sns.heatmap(
        matrix_norm if norm else conf_matrix,
        annot=True,
        ax=axes,
        fmt=".0f",
    )

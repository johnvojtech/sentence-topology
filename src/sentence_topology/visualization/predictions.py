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


def draw_distributions_from_conf_matrix(
    conf_matrix: pd.DataFrame, axes: plt.Axes
) -> None:
    trues = conf_matrix.sum(axis=1)
    preds = conf_matrix.sum(axis=0)
    # hists = pd.DataFrame({"Predicted": preds, "True": trues})
    # sns.lineplot(hists, ax=axes)
    xs = range(len(trues.index))
    axes.bar(xs, trues, label="True", alpha=0.4)
    axes.bar(xs, preds, label="Predicted", alpha=0.4)
    axes.legend()
    axes.set_xticks(xs, labels=trues.index, rotation=90)


def draw_classification_report(report: pd.DataFrame, axes: plt.Axes) -> None:
    report = report.drop("support").stack().reset_index()
    report.columns = ["Metric", "Label", "Value"]
    sns.barplot(report, x="Label", y="Value", hue="Metric")
    axes.set_xticks(axes.get_xticks(), labels=axes.get_xticklabels(), rotation=90)

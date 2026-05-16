from sklearn.metrics import (
    precision_score,
    balanced_accuracy_score,
    recall_score,
    f1_score
)
import pandas as pd
import numpy as np

def calc_model_metrics(y_test: pd.Series, y_pred: np.ndarray) -> pd.DataFrame:
    """
    Calculates model's metrics

    Args:
        y_test: true labels from the dataset (original pandas DataSeries)
        y_pred: Model's prediction labels

    Returns:
        Compact DataFrame
    """

    metrics = {
        "precision macro": precision_score(y_test, y_pred, average="macro"),
        "recall macro": recall_score(y_test, y_pred, average="macro"),
        "f1 macro": f1_score(y_test, y_pred, average="macro"),
        "balanced accuracy": balanced_accuracy_score(y_test, y_pred)
    }

    metrics_df = pd.DataFrame(metrics.items(), columns=["metric", "value"])

    return metrics_df
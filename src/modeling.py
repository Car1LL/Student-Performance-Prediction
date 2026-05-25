from sklearn.metrics import (
    precision_score,
    balanced_accuracy_score,
    recall_score,
    f1_score
)
import pandas as pd
import numpy as np
from typing import Tuple

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

def find_threshold(probs, class_idx, y_true, metric) -> Tuple[float, float]:
    """
    Finds best threshold on a target class, based on the metric

    Args:
        probs: probabilities of a target class, separated from other classes
        class_idx: index of the class, that should be evaluated 
        y_true: true labels
        metric: metric that has to be increased

    Returns:
        Tuple of the best-achieved metric and its threshold value
    """
    
    thresholds = np.arange(0.1, 0.5, 0.01)
    class_probs = probs[:, class_idx]

    results = []

    for t in thresholds:
        y_pred = probs.argmax(axis=1)
        y_pred[class_probs > t] = 0

        score = metric(y_true, y_pred, average="macro")

        results.append((t, score))

    best_t, best_score = max(results, key=lambda x: x[1])

    return best_t, best_score
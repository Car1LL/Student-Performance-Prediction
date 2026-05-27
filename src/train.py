from catboost import CatBoostClassifier, Pool
import pandas as pd
from pathlib import Path
import preprocessing as preprocessing
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from modeling import find_threshold, calc_model_metrics
import matplotlib.pyplot as plt
import joblib
import json
from sklearn.metrics import (recall_score, classification_report, confusion_matrix, 
                             ConfusionMatrixDisplay)



def main():
    df = load_data()
    df = preprocessing.create_class_target_variable(df)
    df = preprocessing.delete_continuous_targets(df)
    X, y = preprocessing.split_X_y(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    cat_features_idx = get_cat_idx(X_train)

    weights = get_weights(y_train)

    train_pool = Pool(
        data=X_train,
        label=y_train,
        cat_features=cat_features_idx
    )

    model = train_model(weights=weights)
    model.fit(train_pool)
    best_threshold = find_best_threshold(model, X_test, y_test)

    probs = model.predict_proba(X_test)
    minor_idx = list(model.classes_).index(0)

    y_pred_final = apply_threshold(
        minor_idx=minor_idx,
        probs=probs,
        best_t=best_threshold
    )

    evaluate(y_pred=y_pred_final, y_test=y_test)

    save_model(model=model, threshold=best_threshold)

def evaluate(y_pred, y_test):
    print(classification_report(y_true=y_test, y_pred=y_pred))
    metric_df = calc_model_metrics(y_pred=y_pred, y_test=y_test)
    print(metric_df)

    model_cm = confusion_matrix(y_pred=y_pred, y_true=y_test)
    ConfusionMatrixDisplay(model_cm).plot()
    plt.title("Confusion Matrix")
    plt.show()


def load_data():
    base_dir = Path(__file__).resolve().parent.parent

    data_dir = (
        base_dir 
        / "dataset"
        / "student"
        / "student-por.csv"
    )

    df = pd.read_csv(data_dir, sep=";")

    return df

def get_cat_idx(X_train):
    cat_features = X_train.select_dtypes(include=['string', 'object']).columns

    cat_features_idx = [X_train.columns.get_loc(col) for col in cat_features]

    return cat_features_idx

def get_weights(y_train):
    classes = np.unique(y_train)

    return compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train
    )


def train_model(weights):
    
    model = CatBoostClassifier(
        loss_function="MultiClass",
        eval_metric="TotalF1",
        class_weights=weights,
        random_state=42,
        logging_level="Silent",
        thread_count=-1,
        depth=6,
        learning_rate=0.05,
        l2_leaf_reg=3,
        iterations=200
    )

    return model

def find_best_threshold(model, X_test, y_test):
    probs = model.predict_proba(X_test)
    minor_idx = list(model.classes_).index(0)

    best_t, _ = find_threshold(
        probs=probs,
        class_idx=minor_idx,
        y_true=y_test,
        metric=recall_score
    )

    return best_t

def apply_threshold(probs, minor_idx, best_t):
    minor_probs = probs[:, minor_idx]

    y_pred_final = probs.argmax(axis=1)

    y_pred_final[minor_probs > best_t] = 0

    return y_pred_final

def save_model(model, threshold):
    base_dir = Path(__file__).resolve().parent.parent
    ARTIFACTS_DIR = base_dir / "artifacts"
    MODEL_PATH = ARTIFACTS_DIR / "model.pkl"
    THRESHOLD_PATH = ARTIFACTS_DIR / "best_threshold.json"

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL_PATH)

    with open(THRESHOLD_PATH, 'w') as f:
        json.dump({
            "threshold": threshold
        }, f)

    print(f"model saved: {MODEL_PATH}")


if __name__ == "__main__":
    main()
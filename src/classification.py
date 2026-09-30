"""Interpretable dispatch-attention classifier using pandas and scikit-learn."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import (
    CLASSIFICATION_TARGET,
    CLASSIFIER_NAME,
    FEATURE_COLUMNS,
    RANDOM_SEED,
    TEST_SIZE,
)
from src.utils import save_joblib, use_plot_backend, write_json

COST_INTERPRETATION = (
    "A false negative is more costly than a false positive in this scenario. "
    "dispatch_attention = 1 means a consignment needs extra checks before trucks "
    "leave Musanze. Missing that flag (false negative) can send an at-risk load "
    "without review. A false positive only adds a staff check, which is cheaper "
    "than a failed dispatch."
)


def _stratified_split(X: pd.DataFrame, y: pd.Series):
    stratify = y if y.value_counts().min() >= 2 else None
    try:
        return train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=RANDOM_SEED,
            stratify=stratify,
        ), stratify is not None
    except ValueError:
        split = train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=RANDOM_SEED,
            stratify=None,
        )
        return split, False


def _plot_confusion(matrix: np.ndarray, path: Path) -> None:
    use_plot_backend()
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Greens",
        cbar=False,
        ax=ax,
        xticklabels=["No attention (0)", "Attention (1)"],
        yticklabels=["No attention (0)", "Attention (1)"],
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Dispatch attention confusion matrix")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_classification(
    df: pd.DataFrame,
    output_dir: Path,
    models_dir: Path,
    group_code: str,
) -> dict:
    X = df[FEATURE_COLUMNS]
    y = df[CLASSIFICATION_TARGET].astype(int)
    (X_train, X_test, y_train, y_test), used_stratify = _stratified_split(X, y)

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    random_state=RANDOM_SEED,
                    max_iter=1000,
                    solver="lbfgs",
                ),
            ),
        ]
    )
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    labels = [0, 1]
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    clf = pipeline.named_steps["clf"]
    coefficient_map = {
        name: float(value) for name, value in zip(FEATURE_COLUMNS, clf.coef_[0])
    }

    metrics = {
        "group_code": group_code,
        "random_seed": RANDOM_SEED,
        "model": CLASSIFIER_NAME,
        "test_size": TEST_SIZE,
        "stratified_split": used_stratify,
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "confusion_matrix": {
            "labels": labels,
            "matrix": cm.tolist(),
            "true_negatives": int(cm[0, 0]),
            "false_positives": int(cm[0, 1]),
            "false_negatives": int(cm[1, 0]),
            "true_positives": int(cm[1, 1]),
        },
        "intercept": float(clf.intercept_[0]),
        "coefficients": coefficient_map,
        "cost_interpretation": COST_INTERPRETATION,
        "note": "The StandardScaler was fitted on the training split only.",
    }
    write_json(output_dir / "classification_metrics.json", metrics)
    _plot_confusion(cm, output_dir / "confusion_matrix.png")
    save_joblib(
        models_dir / "classifier.joblib",
        {
            "pipeline": pipeline,
            "feature_names": FEATURE_COLUMNS,
            "random_seed": RANDOM_SEED,
        },
    )
    print(
        "Classification | "
        f"acc={metrics['accuracy']:.3f} prec={metrics['precision']:.3f} "
        f"rec={metrics['recall']:.3f} f1={metrics['f1']:.3f} "
        f"stratified={used_stratify}"
    )
    print(f"Cost of errors: {COST_INTERPRETATION}")
    return metrics

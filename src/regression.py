"""Linear regression with batch gradient descent implemented in NumPy only.

Hypothesis: h(x) = θ₀ + θ₁x₁ + ... + θₙxₙ
Loss:      J(θ) = (1 / 2m) Σ (h(xᵢ) − yᵢ)²
Update:    θ := θ − α ∇J(θ)
Gradient:  ∇J(θ) = (1 / m) Xᵀ (Xθ − y)

Feature scaling is fitted on the training split only. scikit-learn estimators
are not used in this module.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    FEATURE_COLUMNS,
    LEARNING_RATE,
    N_ITERATIONS,
    RANDOM_SEED,
    REGRESSION_TARGET,
    TEST_SIZE,
)
from src.utils import save_joblib, use_plot_backend, write_json


def train_test_split_numpy(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = X.shape[0]
    if n < 2:
        raise ValueError("Regression needs at least two rows for a train/test split.")
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)
    n_test = max(1, int(round(n * test_size)))
    n_test = min(n_test, n - 1)
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def fit_standardizer(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std = np.where(std == 0.0, 1.0, std)
    return mean, std


def apply_standardizer(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (X - mean) / std


def add_bias(X: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(X.shape[0], dtype=float), X])


def batch_gradient_descent(
    X_b: np.ndarray,
    y: np.ndarray,
    learning_rate: float,
    n_iterations: int,
) -> tuple[np.ndarray, list[float]]:
    m, n_params = X_b.shape
    theta = np.zeros(n_params, dtype=float)
    history: list[float] = []
    for _ in range(n_iterations):
        error = X_b @ theta - y
        loss = float((error ** 2).sum() / (2.0 * m))
        history.append(loss)
        gradient = (X_b.T @ error) / m
        theta = theta - learning_rate * gradient
    return theta, history


def predict_linear(X_b: np.ndarray, theta: np.ndarray) -> np.ndarray:
    return X_b @ theta


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    residual = float(np.sum((y_true - y_pred) ** 2))
    total = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if total == 0.0:
        return 0.0
    return float(1.0 - residual / total)


def _plot_loss(history: list[float], path: Path) -> None:
    use_plot_backend()
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(np.arange(1, len(history) + 1), history, color="#1B5E20", linewidth=2)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Training loss J(θ)")
    ax.set_title("Batch gradient descent loss curve")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_regression(
    df: pd.DataFrame,
    output_dir: Path,
    models_dir: Path,
    group_code: str,
) -> dict:
    X = df[FEATURE_COLUMNS].to_numpy(dtype=float)
    y = df[REGRESSION_TARGET].to_numpy(dtype=float)
    X_train, X_test, y_train, y_test = train_test_split_numpy(
        X, y, TEST_SIZE, RANDOM_SEED
    )
    mean, std = fit_standardizer(X_train)
    X_train_s = apply_standardizer(X_train, mean, std)
    X_test_s = apply_standardizer(X_test, mean, std)
    X_train_b = add_bias(X_train_s)
    X_test_b = add_bias(X_test_s)

    theta, history = batch_gradient_descent(
        X_train_b, y_train, LEARNING_RATE, N_ITERATIONS
    )
    y_pred = predict_linear(X_test_b, theta)
    if not np.isfinite(y_pred).all():
        raise RuntimeError("Regression produced non-finite predictions.")

    metrics = {
        "group_code": group_code,
        "random_seed": RANDOM_SEED,
        "learning_rate": LEARNING_RATE,
        "n_iterations": N_ITERATIONS,
        "test_size": TEST_SIZE,
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "mae": mean_absolute_error(y_test, y_pred),
        "rmse": root_mean_squared_error(y_test, y_pred),
        "r_squared": r_squared(y_test, y_pred),
        "final_train_loss": history[-1],
        "initial_train_loss": history[0],
        "loss_decreased": bool(history[-1] < history[0]),
        "coefficients": {
            "intercept": float(theta[0]),
            **{
                name: float(value)
                for name, value in zip(FEATURE_COLUMNS, theta[1:])
            },
        },
        "test_predictions": [float(v) for v in y_pred],
        "test_actual": [float(v) for v in y_test],
        "note": (
            "Weights come from NumPy batch gradient descent. "
            "The scaler was fitted on training features only."
        ),
    }
    write_json(output_dir / "regression_metrics.json", metrics)
    _plot_loss(history, output_dir / "regression_loss.png")
    save_joblib(
        models_dir / "regression.joblib",
        {
            "theta": theta,
            "mean": mean,
            "std": std,
            "feature_names": FEATURE_COLUMNS,
            "learning_rate": LEARNING_RATE,
            "n_iterations": N_ITERATIONS,
            "random_seed": RANDOM_SEED,
        },
    )
    print(
        "Regression | "
        f"MAE={metrics['mae']:.3f} RMSE={metrics['rmse']:.3f} "
        f"R²={metrics['r_squared']:.3f} loss_start={history[0]:.3f} "
        f"loss_end={history[-1]:.3f}"
    )
    return metrics

"""Unsupervised farm-profile clustering on input features only."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from src.config import FEATURE_COLUMNS, ID_COLUMN, K_MAX, K_MIN, RANDOM_SEED
from src.utils import save_joblib, use_plot_backend, write_json

INTERPRETATION_NOTE = (
    "Cluster labels describe similar input-feature profiles in this file. "
    "They are not verified real-world farm categories, risk grades, or "
    "operational types. Staff should treat them as a starting grouping, "
    "not as a confirmed identity for any collection point."
)


def _max_k(n_samples: int) -> int:
    return min(K_MAX, n_samples - 1)


def evaluate_k(X_scaled: np.ndarray) -> dict[int, float | None]:
    n_samples = X_scaled.shape[0]
    upper = _max_k(n_samples)
    scores: dict[int, float | None] = {}
    for k in range(K_MIN, K_MAX + 1):
        if k > upper or k < 2:
            scores[k] = None
            continue
        model = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
        labels = model.fit_predict(X_scaled)
        if len(set(labels)) < 2:
            scores[k] = None
            continue
        scores[k] = float(silhouette_score(X_scaled, labels))
    return scores


def select_k(scores: dict[int, float | None]) -> int:
    valid = {k: score for k, score in scores.items() if score is not None}
    if not valid:
        raise RuntimeError(
            "Could not compute a silhouette score for any k in 2..5. "
            "The CSV needs more rows than the requested k values."
        )
    return max(valid, key=valid.get)


def _plot_clusters(X_scaled: np.ndarray, labels: np.ndarray, path: Path) -> None:
    use_plot_backend()
    import matplotlib.pyplot as plt
    import seaborn as sns

    n_components = min(2, X_scaled.shape[1], X_scaled.shape[0])
    if n_components < 2:
        coords = np.column_stack(
            [X_scaled[:, 0], np.zeros(X_scaled.shape[0])]
        )
        x_label, y_label = "Feature 1 (scaled)", "Placeholder"
    else:
        coords = PCA(n_components=2, random_state=RANDOM_SEED).fit_transform(X_scaled)
        x_label, y_label = "PCA 1", "PCA 2"

    plot_df = pd.DataFrame(
        {"x": coords[:, 0], "y": coords[:, 1], "cluster": labels.astype(str)}
    )
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8, 5.5))
    sns.scatterplot(
        data=plot_df,
        x="x",
        y="y",
        hue="cluster",
        palette="Set2",
        s=80,
        ax=ax,
    )
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title("Collection-point profiles (PCA view of clusters)")
    ax.legend(title="Cluster")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_clustering(
    df: pd.DataFrame,
    output_dir: Path,
    models_dir: Path,
    group_code: str,
) -> dict:
    features = df[FEATURE_COLUMNS]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)
    scores = evaluate_k(X_scaled)
    chosen_k = select_k(scores)
    model = KMeans(n_clusters=chosen_k, random_state=RANDOM_SEED, n_init=10)
    labels = model.fit_predict(X_scaled)

    cluster_table = pd.DataFrame(
        {
            ID_COLUMN: df[ID_COLUMN].astype(str).to_numpy(),
            "cluster_label": labels.astype(int),
        }
    )
    cluster_table.to_csv(output_dir / "clusters.csv", index=False)

    chosen_score = scores[chosen_k]
    justification = (
        f"k={chosen_k} was selected because it has the highest silhouette "
        f"score ({chosen_score:.4f}) among valid k values from {K_MIN} to {K_MAX}. "
        "Silhouette close to 1 means tighter, better-separated groups; values "
        "near 0 mean overlapping groups. This is a compactness score, not proof "
        "that the groups exist in the field."
    )
    metrics = {
        "group_code": group_code,
        "random_seed": RANDOM_SEED,
        "k_range": list(range(K_MIN, K_MAX + 1)),
        "silhouette_scores": {str(k): score for k, score in scores.items()},
        "selected_k": int(chosen_k),
        "selected_silhouette": chosen_score,
        "n_records_labeled": int(len(labels)),
        "cluster_sizes": {
            str(cluster): int((labels == cluster).sum())
            for cluster in sorted(set(labels.tolist()))
        },
        "justification": justification,
        "interpretation_note": INTERPRETATION_NOTE,
        "features_used": FEATURE_COLUMNS,
        "targets_excluded": ["actual_yield_kg", "dispatch_attention", "record_id"],
    }
    write_json(output_dir / "clustering_metrics.json", metrics)
    _plot_clusters(X_scaled, labels, output_dir / "cluster_plot.png")
    save_joblib(
        models_dir / "cluster.joblib",
        {
            "scaler": scaler,
            "model": model,
            "feature_names": FEATURE_COLUMNS,
            "selected_k": int(chosen_k),
            "random_seed": RANDOM_SEED,
        },
    )
    print(
        "Clustering | "
        f"selected_k={chosen_k} silhouette={chosen_score:.4f} "
        f"labels={len(labels)}"
    )
    print(f"Clustering caution: {INTERPRETATION_NOTE}")
    return metrics

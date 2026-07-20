from pathlib import Path
from typing import Dict, Sequence, Tuple
import argparse
import numpy as np


def get_class_colors(num_labels: int) -> Sequence[str]:
    palette = [
        "#1f77b4",
        "#d62728",
        "#2ca02c",
        "#ff7f0e",
        "#9467bd",
        "#17becf",
        "#e377c2",
        "#8c564b",
        "#bcbd22",
        "#7f7f7f",
    ]
    return palette[:num_labels]


def compute_projection(
    method: str,
    embeddings: np.ndarray,
    labels_np: np.ndarray,
    seed: int,
) -> Tuple[np.ndarray, float]:
    try:
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE
        from sklearn.metrics import silhouette_score
    except Exception as exc:
        raise ImportError(
            "Embedding analysis requires scikit-learn for PCA, t-SNE, and silhouette score."
        ) from exc

    if len(embeddings) < 2:
        projection = np.zeros((len(embeddings), 2), dtype=np.float32)
    elif method == "pca":
        projection = PCA(n_components=2, random_state=seed).fit_transform(embeddings)
    elif method == "tsne":
        if len(embeddings) < 4:
            projection = PCA(n_components=2, random_state=seed).fit_transform(embeddings)
        else:
            perplexity = min(30, max(2, (len(embeddings) - 1) // 3))
            perplexity = min(perplexity, len(embeddings) - 1)
            projection = TSNE(
                n_components=2,
                perplexity=perplexity,
                init="pca",
                learning_rate="auto",
                random_state=seed,
            ).fit_transform(embeddings)
    elif method == "umap":
        if len(embeddings) < 4:
            projection = PCA(n_components=2, random_state=seed).fit_transform(embeddings)
        else:
            try:
                import umap
            except Exception as exc:
                raise ImportError("Embedding analysis requires umap-learn for UMAP visualization.") from exc
            n_neighbors = min(15, max(2, len(embeddings) - 1))
            projection = umap.UMAP(
                n_components=2,
                n_neighbors=n_neighbors,
                min_dist=0.1,
                metric="cosine",
                random_state=seed,
            ).fit_transform(embeddings)
    else:
        raise ValueError(f"Unknown projection method: {method}")

    unique_labels = np.unique(labels_np)
    if len(unique_labels) < 2 or len(unique_labels) >= len(labels_np):
        silhouette = float("nan")
    else:
        silhouette = float(silhouette_score(embeddings, labels_np, metric="cosine"))
    return projection, silhouette


def add_color_key(fig, labels: Sequence[str], colors: Sequence[str]) -> None:
    fig.text(0.06, 0.018, "Class colors:", ha="left", va="center", fontsize=14, fontweight="bold")
    x = 0.17
    step = 0.80 / max(len(labels), 1)
    for label, color in zip(labels, colors):
        fig.text(x, 0.018, label, ha="left", va="center", fontsize=14, color=color, fontweight="bold")
        x += step


def save_embedding_analysis(
    split_payloads: Dict[str, Tuple[Sequence, np.ndarray, np.ndarray]],
    labels: Sequence[str],
    args: argparse.Namespace,
) -> Path:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise ImportError("Embedding analysis requires matplotlib.") from exc

    methods = [("tsne", "t-SNE"), ("pca", "PCA"), ("umap", "UMAP")]
    split_order = ["train", "valid", "test"]
    colors = get_class_colors(len(labels))
    fig, axes = plt.subplots(len(split_order), len(methods), figsize=(27, 24))
    row_silhouettes: dict[str, float] = {}

    for row_idx, split_name in enumerate(split_order):
        samples, embeddings, y = split_payloads[split_name]
        x_np = embeddings.numpy()
        x_np = x_np / np.clip(np.linalg.norm(x_np, axis=1, keepdims=True), 1e-12, None)
        y_np = y.numpy()
        for col_idx, (method_key, method_title) in enumerate(methods):
            ax = axes[row_idx, col_idx]
            projection, silhouette = compute_projection(method_key, x_np, y_np, args.seed)
            row_silhouettes.setdefault(split_name, silhouette)
            for label_idx, label_name in enumerate(labels):
                mask = y_np == label_idx
                if mask.any():
                    ax.scatter(
                        projection[mask, 0],
                        projection[mask, 1],
                        s=13,
                        alpha=0.78,
                        color=colors[label_idx],
                        linewidths=0,
                    )
            ax.set_title(f"{split_name.upper()} - {method_title}", fontsize=18, fontweight="bold")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.grid(False)

    fig.subplots_adjust(left=0.035, right=0.985, top=0.955, bottom=0.08, hspace=0.26, wspace=0.055)
    for row_idx, split_name in enumerate(split_order):
        row_axes = axes[row_idx, :]
        bottom = min(ax.get_position().y0 for ax in row_axes)
        silhouette = row_silhouettes.get(split_name, float("nan"))
        silhouette_text = "nan" if np.isnan(silhouette) else f"{silhouette:.4f}"
        fig.text(
            0.5,
            bottom - 0.018,
            f"{split_name.upper()} silhouette score: {silhouette_text}",
            ha="center",
            va="top",
            fontsize=16,
            fontweight="bold",
        )
    add_color_key(fig, labels, colors)

    output_dir = Path("/scratch/data/bikash_rs/Vivek/PRC-Emo/analysis/embed_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{args.model_tag}_{args.dataset}_mlp_{args.num_labels}cls_{args.seed}_embedding_analysis.png"
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved embedding analysis visualization to {output_path}")
    return output_path

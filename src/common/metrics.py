from typing import Dict, Sequence

import numpy as np

try:
    from sklearn.metrics import classification_report as sklearn_classification_report, confusion_matrix as sklearn_confusion_matrix
except Exception:
    sklearn_classification_report = None
    sklearn_confusion_matrix = None


def make_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_labels: int) -> np.ndarray:
    if sklearn_confusion_matrix is not None:
        return sklearn_confusion_matrix(y_true, y_pred, labels=list(range(num_labels)))
    matrix = np.zeros((num_labels, num_labels), dtype=int)
    for true_id, pred_id in zip(y_true, y_pred):
        if 0 <= int(true_id) < num_labels and 0 <= int(pred_id) < num_labels:
            matrix[int(true_id), int(pred_id)] += 1
    return matrix


def make_classification_report(y_true: np.ndarray, y_pred: np.ndarray, labels: Sequence[str]) -> str:
    if sklearn_classification_report is not None:
        return sklearn_classification_report(
            y_true,
            y_pred,
            labels=list(range(len(labels))),
            target_names=list(labels),
            digits=4,
            zero_division=0,
        )

    rows = []
    rows.append("scikit-learn unavailable; using fallback metrics.")
    rows.append(f"{'label':>14} {'precision':>10} {'recall':>10} {'f1-score':>10} {'support':>10}")
    f1_values = []
    supports = []
    for idx, label in enumerate(labels):
        tp = int(((y_true == idx) & (y_pred == idx)).sum())
        fp = int(((y_true != idx) & (y_pred == idx)).sum())
        fn = int(((y_true == idx) & (y_pred != idx)).sum())
        support = int((y_true == idx).sum())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        f1_values.append(f1)
        supports.append(support)
        rows.append(f"{label:>14} {precision:10.4f} {recall:10.4f} {f1:10.4f} {support:10d}")
    accuracy = float((y_true == y_pred).mean()) if len(y_true) else 0.0
    macro_f1 = float(np.mean(f1_values)) if f1_values else 0.0
    weighted_f1 = float(np.average(f1_values, weights=supports)) if sum(supports) else 0.0
    rows.append("")
    rows.append(f"{'accuracy':>14} {'':>10} {'':>10} {accuracy:10.4f} {len(y_true):10d}")
    rows.append(f"{'macro avg':>14} {'':>10} {'':>10} {macro_f1:10.4f} {len(y_true):10d}")
    rows.append(f"{'weighted avg':>14} {'':>10} {'':>10} {weighted_f1:10.4f} {len(y_true):10d}")
    return "\n".join(rows)

"""Threshold selection, binary classification metrics and bootstrap confidence intervals."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve


def choose_threshold(
    y_true: np.ndarray, y_prob: np.ndarray, strategy: str = "sensitivity", target_sensitivity: float = 0.85
) -> float:
    """Pick a decision threshold on *validation* data.

    ``sensitivity``: highest threshold whose sensitivity is >= target (screening use case).
    ``youden``: threshold maximising sensitivity + specificity - 1.
    """
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    thr = np.clip(thr, 0.0, 1.0)  # sklearn prepends +inf
    if strategy == "youden":
        return float(thr[np.argmax(tpr - fpr)])
    if strategy == "sensitivity":
        ok = np.where(tpr >= target_sensitivity)[0]
        return float(thr[ok[0]])
    raise ValueError(f"unknown threshold strategy {strategy!r}")


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, int]:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    return {
        "tp": int(((y_true == 1) & (y_pred == 1)).sum()),
        "fp": int(((y_true == 0) & (y_pred == 1)).sum()),
        "tn": int(((y_true == 0) & (y_pred == 0)).sum()),
        "fn": int(((y_true == 1) & (y_pred == 0)).sum()),
    }


def _safe_div(a: float, b: float) -> float:
    return float(a / b) if b else float("nan")


def binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict[str, float]:
    """All metrics are computed from the same predictions at the same threshold."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    c = confusion_counts(y_true, y_prob >= threshold)
    sens = _safe_div(c["tp"], c["tp"] + c["fn"])
    spec = _safe_div(c["tn"], c["tn"] + c["fp"])
    ppv = _safe_div(c["tp"], c["tp"] + c["fp"])
    both = len(np.unique(y_true)) == 2
    return {
        "auroc": float(roc_auc_score(y_true, y_prob)) if both else float("nan"),
        "auprc": float(average_precision_score(y_true, y_prob)) if both else float("nan"),
        "sensitivity": sens,
        "specificity": spec,
        "ppv": ppv,
        "npv": _safe_div(c["tn"], c["tn"] + c["fn"]),
        "f1": _safe_div(2 * ppv * sens, ppv + sens),
        "balanced_accuracy": (sens + spec) / 2,
        "accuracy": _safe_div(c["tp"] + c["tn"], len(y_true)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "threshold": float(threshold),
        "n": int(len(y_true)),
        "prevalence": float(y_true.mean()),
        **c,
    }


def bootstrap_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    n_boot: int = 1000,
    groups: np.ndarray | None = None,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[str, tuple[float, float]]:
    """Percentile bootstrap CIs. If ``groups`` (e.g. lesion ids) is given, resample whole
    groups so that correlated images of the same lesion are not treated as independent."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    if groups is None:
        index_sets = [np.array([i]) for i in range(len(y_true))]
    else:
        groups = np.asarray(groups)
        _, inv = np.unique(groups, return_inverse=True)
        index_sets = [np.where(inv == g)[0] for g in range(inv.max() + 1)]

    samples: dict[str, list[float]] = {}
    for _ in range(n_boot):
        pick = rng.integers(0, len(index_sets), len(index_sets))
        idx = np.concatenate([index_sets[i] for i in pick])
        if len(np.unique(y_true[idx])) < 2:
            continue
        m = binary_metrics(y_true[idx], y_prob[idx], threshold)
        for k in ("auroc", "auprc", "sensitivity", "specificity", "ppv", "npv", "f1", "balanced_accuracy"):
            samples.setdefault(k, []).append(m[k])
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    return {k: (float(np.nanpercentile(v, lo)), float(np.nanpercentile(v, hi))) for k, v in samples.items()}

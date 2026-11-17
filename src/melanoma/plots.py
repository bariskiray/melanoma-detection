"""Evaluation figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve  # noqa: E402

COLORS = ["#2563eb", "#ea580c", "#16a34a", "#9333ea"]


def _finish(fig, path: Path) -> None:
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def roc_pr_curves(curves: dict[str, tuple[np.ndarray, np.ndarray]], path: Path) -> None:
    """``curves`` maps a legend label to (y_true, y_prob)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6))
    for color, (name, (y, p)) in zip(COLORS, curves.items(), strict=False):
        fpr, tpr, _ = roc_curve(y, p)
        ax1.plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUROC {roc_auc_score(y, p):.3f})")
        prec, rec, _ = precision_recall_curve(y, p)
        ax2.plot(rec, prec, color=color, lw=2, label=f"{name} (AUPRC {average_precision_score(y, p):.3f})")
        ax2.axhline(y.mean(), color=color, lw=1, ls=":", alpha=0.7)
    ax1.plot([0, 1], [0, 1], color="grey", ls="--", lw=1)
    ax1.set(xlabel="1 - specificity", ylabel="Sensitivity", title="ROC")
    ax2.set(xlabel="Recall (sensitivity)", ylabel="Precision (PPV)", title="Precision-recall (dotted = prevalence)")
    for ax in (ax1, ax2):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.3)
        ax.legend(loc="lower right" if ax is ax1 else "upper right", fontsize=8)
    _finish(fig, path)


def confusion(counts: dict[str, int], title: str, path: Path) -> None:
    cm = np.array([[counts["tn"], counts["fp"]], [counts["fn"], counts["tp"]]])
    fig, ax = plt.subplots(figsize=(4.4, 3.9))
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, f"{v}", ha="center", va="center", fontsize=13, color="white" if v > cm.max() / 2 else "black")
    ax.set_xticks([0, 1], ["Benign", "Melanoma"])
    ax.set_yticks([0, 1], ["Benign", "Melanoma"])
    ax.set(xlabel="Predicted", ylabel="Actual", title=title)
    _finish(fig, path)


def calibration(curves: dict[str, tuple[np.ndarray, np.ndarray]], path: Path, n_bins: int = 10) -> None:
    fig, ax = plt.subplots(figsize=(4.8, 4.4))
    ax.plot([0, 1], [0, 1], color="grey", ls="--", lw=1, label="Perfect calibration")
    for color, (name, (y, p)) in zip(COLORS, curves.items(), strict=False):
        frac, mean = calibration_curve(y, p, n_bins=n_bins, strategy="quantile")
        ax.plot(mean, frac, marker="o", color=color, lw=1.5, label=name)
    ax.set(xlabel="Predicted probability", ylabel="Observed melanoma fraction", title="Calibration")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    _finish(fig, path)


def training_history(history: pd.DataFrame, path: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))
    ax1.plot(history["epoch"], history["train_loss"], label="train", color=COLORS[0])
    ax1.plot(history["epoch"], history["val_loss"], label="validation", color=COLORS[1])
    ax1.set(xlabel="Epoch", ylabel="BCE loss", title="Loss")
    ax2.plot(history["epoch"], history["val_auroc"], color=COLORS[1])
    best = history["val_auroc"].idxmax()
    ax2.scatter(history.loc[best, "epoch"], history.loc[best, "val_auroc"], color="black", zorder=3, label="selected")
    ax2.set(xlabel="Epoch", ylabel="AUROC", title="Validation AUROC")
    for ax in (ax1, ax2):
        ax.grid(alpha=0.3)
        ax.legend()
    _finish(fig, path)

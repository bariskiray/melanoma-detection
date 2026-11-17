import numpy as np
import pytest

from melanoma.metrics import binary_metrics, bootstrap_ci, choose_threshold, confusion_counts


def test_confusion_counts():
    c = confusion_counts(np.array([1, 1, 0, 0, 1]), np.array([1, 0, 0, 1, 1]))
    assert c == {"tp": 2, "fp": 1, "tn": 1, "fn": 1}


def test_metrics_are_consistent_with_confusion_matrix():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    p = np.clip(y * 0.3 + rng.random(500) * 0.7, 0, 1)
    m = binary_metrics(y, p, 0.4)
    assert m["sensitivity"] == pytest.approx(m["tp"] / (m["tp"] + m["fn"]))
    assert m["specificity"] == pytest.approx(m["tn"] / (m["tn"] + m["fp"]))
    assert m["ppv"] == pytest.approx(m["tp"] / (m["tp"] + m["fp"]))
    assert m["accuracy"] == pytest.approx((m["tp"] + m["tn"]) / 500)
    assert m["tp"] + m["fp"] + m["tn"] + m["fn"] == 500


def test_perfect_classifier():
    y = np.array([0, 0, 1, 1])
    m = binary_metrics(y, np.array([0.1, 0.2, 0.8, 0.9]), 0.5)
    assert m["auroc"] == 1.0 and m["sensitivity"] == 1.0 and m["specificity"] == 1.0


def test_sensitivity_threshold_reaches_target():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 1000)
    p = np.clip(y * 0.25 + rng.random(1000) * 0.75, 0, 1)
    t = choose_threshold(y, p, "sensitivity", 0.9)
    assert binary_metrics(y, p, t)["sensitivity"] >= 0.9


def test_youden_threshold_in_range():
    y = np.array([0, 0, 0, 1, 1, 1])
    p = np.array([0.1, 0.3, 0.45, 0.4, 0.7, 0.9])
    assert 0.0 <= choose_threshold(y, p, "youden") <= 1.0


def test_unknown_threshold_strategy():
    with pytest.raises(ValueError):
        choose_threshold(np.array([0, 1]), np.array([0.2, 0.8]), "magic")


@pytest.mark.parametrize("grouped", [False, True])
def test_bootstrap_ci_brackets_point_estimate(grouped):
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, 400)
    p = np.clip(y * 0.3 + rng.random(400) * 0.7, 0, 1)
    groups = np.repeat(np.arange(200), 2) if grouped else None
    ci = bootstrap_ci(y, p, 0.5, n_boot=200, groups=groups)
    auc = binary_metrics(y, p, 0.5)["auroc"]
    lo, hi = ci["auroc"]
    assert lo <= auc <= hi

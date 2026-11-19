"""Discrimination and calibration metrics, with bootstrap confidence intervals."""

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve


def ks_statistic(y, p) -> float:
    """Kolmogorov-Smirnov: largest gap between the score distributions of bads and goods."""
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


def all_metrics(y, p) -> dict:
    y, p = np.asarray(y), np.asarray(p)
    auc = roc_auc_score(y, p)
    return {
        "auc": float(auc),
        "gini": float(2 * auc - 1),
        "ks": ks_statistic(y, p),
        "brier": float(brier_score_loss(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
    }


def bootstrap_metrics(y, p, n_boot=500, seed=0, alpha=0.05) -> dict:
    """Point estimate and percentile interval for each metric (resampling loans with replacement)."""
    y, p = np.asarray(y), np.asarray(p)
    rng = np.random.default_rng(seed)
    point = all_metrics(y, p)
    draws = {k: [] for k in point}
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        for k, v in all_metrics(y[idx], p[idx]).items():
            draws[k].append(v)
    out = {}
    for k, v in point.items():
        lo, hi = np.quantile(draws[k], [alpha / 2, 1 - alpha / 2])
        out[k] = {"value": v, "low": float(lo), "high": float(hi)}
    return out


def paired_auc_difference(y, p1, p2, n_boot=500, seed=0) -> dict:
    """Bootstrap interval of AUC(p1) - AUC(p2) on the same resampled loans."""
    y, p1, p2 = np.asarray(y), np.asarray(p1), np.asarray(p2)
    rng = np.random.default_rng(seed)
    n = len(y)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs.append(roc_auc_score(y[idx], p1[idx]) - roc_auc_score(y[idx], p2[idx]))
    d = roc_auc_score(y, p1) - roc_auc_score(y, p2)
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    return {"diff": float(d), "low": float(lo), "high": float(hi)}


def calibration_bins(y, p, n_bins=10):
    """Mean predicted PD and observed default rate in equal-count bins of the prediction."""
    y, p = np.asarray(y), np.asarray(p)
    order = np.argsort(p)
    groups = np.array_split(order, n_bins)
    pred = np.array([p[g].mean() for g in groups])
    obs = np.array([y[g].mean() for g in groups])
    return pred, obs


def expected_calibration_error(y, p, n_bins=10) -> float:
    pred, obs = calibration_bins(y, p, n_bins)
    return float(np.mean(np.abs(pred - obs)))

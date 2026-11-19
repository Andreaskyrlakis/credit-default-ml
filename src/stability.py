"""Population stability index (PSI) and performance by sub-group."""

import numpy as np
import pandas as pd

from .metrics import all_metrics


def psi(expected, actual, n_bins=10, eps=1e-4) -> float:
    """PSI = sum (a - e) ln(a / e) over bins cut at the deciles of the expected (reference) sample.

    Usual reading: < 0.10 stable, 0.10-0.25 moderate shift, > 0.25 significant shift.
    """
    expected, actual = np.asarray(expected, float), np.asarray(actual, float)
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, n_bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)
    e, a = np.clip(e, eps, None), np.clip(a, eps, None)
    return float(np.sum((a - e) * np.log(a / e)))


def psi_categorical(expected, actual, eps=1e-4) -> float:
    e = pd.Series(expected).astype("string").value_counts(normalize=True)
    a = pd.Series(actual).astype("string").value_counts(normalize=True)
    cats = e.index.union(a.index)
    e = e.reindex(cats, fill_value=0).clip(lower=eps)
    a = a.reindex(cats, fill_value=0).clip(lower=eps)
    return float(np.sum((a - e) * np.log(a / e)))


def performance_by_group(df: pd.DataFrame, group: str, y="default", p="pd", min_n=500):
    """AUC, observed and predicted default rate in each group (groups smaller than min_n dropped)."""
    rows = []
    for g, d in df.groupby(group, observed=True):
        if len(d) < min_n or d[y].nunique() < 2:
            continue
        m = all_metrics(d[y], d[p])
        rows.append({"group": str(g), "n": len(d), "observed_dr": d[y].mean(),
                     "predicted_pd": d[p].mean(), "auc": m["auc"]})
    return pd.DataFrame(rows)

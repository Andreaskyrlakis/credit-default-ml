"""Binning, weight of evidence (WoE), information value (IV) and the points of a scorecard.

Convention: WoE = ln(share of goods / share of bads) in the bin, so a positive WoE means a safer
bin. IV = sum over bins of (share of goods - share of bads) * WoE.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

SMOOTH = 0.5  # added to good and bad counts so that a pure bin has a finite WoE


def woe_table(bins: pd.Series, y: pd.Series) -> pd.DataFrame:
    """Counts, default rate, WoE and IV contribution for each bin label."""
    t = pd.crosstab(bins, y)
    t = t.reindex(columns=[0, 1], fill_value=0)
    t.columns = ["good", "bad"]
    good = t["good"] + SMOOTH
    bad = t["bad"] + SMOOTH
    t["n"] = t["good"] + t["bad"]
    t["default_rate"] = t["bad"] / t["n"]
    t["woe"] = np.log((good / good.sum()) / (bad / bad.sum()))
    t["iv"] = (good / good.sum() - bad / bad.sum()) * t["woe"]
    return t


def _monotone_edges(x: pd.Series, y: pd.Series, max_bins: int) -> np.ndarray:
    """Quantile edges, then merge neighbours until the default rate is monotone in x."""
    edges = np.unique(np.nanquantile(x, np.linspace(0, 1, max_bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    while len(edges) > 2:
        b = pd.cut(x, edges, include_lowest=True)
        rate = y.groupby(b, observed=False).mean().to_numpy()
        diffs = np.diff(rate)
        sign = np.sign(np.nansum(diffs))
        bad = np.where(np.sign(diffs) == -sign)[0] if sign != 0 else np.array([], dtype=int)
        if len(bad) == 0:
            break
        # drop the inner edge between the two bins that break monotonicity
        edges = np.delete(edges, bad[0] + 1)
    return edges


@dataclass
class FeatureBinning:
    name: str
    kind: str                      # "numeric" or "categorical"
    edges: np.ndarray | None = None
    groups: dict = field(default_factory=dict)   # category -> bin label
    woe: dict = field(default_factory=dict)      # bin label -> WoE
    table: pd.DataFrame | None = None

    @property
    def iv(self) -> float:
        return float(self.table["iv"].sum())

    def labels(self, x: pd.Series) -> pd.Series:
        if self.kind == "numeric":
            lab = pd.cut(x, self.edges, include_lowest=True).astype("string")
            return lab.fillna("Missing")
        return x.astype("string").map(self.groups).fillna("Other")

    def transform(self, x: pd.Series) -> np.ndarray:
        lab = self.labels(x)
        return lab.map(self.woe).fillna(0.0).to_numpy(dtype=float)


def fit_binning(x: pd.Series, y: pd.Series, name: str, max_bins: int = 10,
                min_share: float = 0.01) -> FeatureBinning:
    if pd.api.types.is_numeric_dtype(x):
        fb = FeatureBinning(name, "numeric", edges=_monotone_edges(x, y, max_bins))
    else:
        share = x.astype("string").value_counts(normalize=True)
        groups = {c: (c if s >= min_share else "Other") for c, s in share.items()}
        fb = FeatureBinning(name, "categorical", groups=groups)
    lab = fb.labels(x)
    table = woe_table(lab, y)
    if fb.kind == "numeric":
        # crosstab sorts labels as text ("(100000" before "(31000"): restore the numeric order
        order = [str(iv) for iv in pd.cut(pd.Series([0.0]), fb.edges).cat.categories]
        table = table.reindex([o for o in order + ["Missing"] if o in table.index])
    fb.table = table
    fb.woe = fb.table["woe"].to_dict()
    return fb


class WoEScorecard:
    """Logistic regression on WoE-transformed features (the classic bank application scorecard)."""

    def __init__(self, features, min_iv=0.02, max_bins=10, C=1.0):
        self.features, self.min_iv, self.max_bins, self.C = list(features), min_iv, max_bins, C

    def fit(self, X: pd.DataFrame, y):
        y = pd.Series(np.asarray(y), index=X.index)
        self.binnings = {f: fit_binning(X[f], y, f, self.max_bins) for f in self.features}
        self.iv = pd.Series({f: b.iv for f, b in self.binnings.items()}).sort_values(ascending=False)
        self.selected = [f for f in self.iv.index if self.iv[f] >= self.min_iv]
        self.model = LogisticRegression(C=self.C, max_iter=1000)
        self.model.fit(self.woe_matrix(X), y)
        self.classes_ = self.model.classes_
        return self

    def woe_matrix(self, X: pd.DataFrame) -> np.ndarray:
        return np.column_stack([self.binnings[f].transform(X[f]) for f in self.selected])

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(self.woe_matrix(X))

    def points_table(self, pdo=20.0, base_score=600.0, base_odds=50.0) -> pd.DataFrame:
        """Points per bin so that score = sum of points; +pdo points doubles the good:bad odds.

        The model gives ln(bad/good) = b0 + sum_j b_j WoE_j, i.e. ln(good/bad) = -(...).
        Score = offset + factor * ln(good/bad), split evenly across features for the intercept.
        """
        factor = pdo / np.log(2.0)
        offset = base_score - factor * np.log(base_odds)
        b0 = self.model.intercept_[0]
        n = len(self.selected)
        rows = []
        for f, b in zip(self.selected, self.model.coef_[0]):
            for label, w in self.binnings[f].woe.items():
                pts = -(b * w + b0 / n) * factor + offset / n
                rows.append({"feature": f, "bin": label, "woe": w, "points": pts})
        return pd.DataFrame(rows)

    def score(self, X: pd.DataFrame, pdo=20.0, base_score=600.0, base_odds=50.0) -> np.ndarray:
        factor = pdo / np.log(2.0)
        offset = base_score - factor * np.log(base_odds)
        log_odds_bad = self.model.decision_function(self.woe_matrix(X))
        return offset - factor * log_odds_bad

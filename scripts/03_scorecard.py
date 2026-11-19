"""Baseline: WoE binning, information value and a logistic scorecard (points), plus the
Lending Club sub-grade as an external benchmark.

Output: results/scorecard.json, results/scorecard_points.csv, figures 3-4.
"""

import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src import config  # noqa: E402
from src.data import load_loans, modelling_sample, time_split  # noqa: E402
from src.metrics import all_metrics  # noqa: E402
from src.plotting import savefig, style  # noqa: E402
from src.woe import WoEScorecard  # noqa: E402

style()
train, valid, test = time_split(modelling_sample(load_loans()))
y = config.TARGET

sc = WoEScorecard(config.FEATURES, min_iv=0.02).fit(train[config.FEATURES], train[y])
pd_valid = sc.predict_proba(valid[config.FEATURES])[:, 1]
pd_test = sc.predict_proba(test[config.FEATURES])[:, 1]
points = sc.points_table()
points.to_csv(config.RESULTS / "scorecard_points.csv", index=False)
pickle.dump(sc, open(config.PROCESSED / "scorecard.pkl", "wb"))

# Lending Club's own sub-grade (A1 ... G5) as a benchmark PD: its default rate in the training set
grade_pd = train.groupby(train["sub_grade"].astype(str))[y].mean()
lc_test = test["sub_grade"].astype(str).map(grade_pd).fillna(train[y].mean()).to_numpy()

score_test = sc.score(test[config.FEATURES])
out = {
    "iv": sc.iv.round(4).to_dict(),
    "selected": sc.selected,
    "coefficients": dict(zip(sc.selected, np.round(sc.model.coef_[0], 4).tolist())),
    "valid": all_metrics(valid[y], pd_valid),
    "test": all_metrics(test[y], pd_test),
    "lc_subgrade_test": all_metrics(test[y], lc_test),
    "score_range_test": [float(np.percentile(score_test, 1)), float(np.percentile(score_test, 99))],
    "n_bins": {f: int(len(sc.binnings[f].table)) for f in sc.selected},
}
json.dump(out, open(config.RESULTS / "scorecard.json", "w"), indent=2)
print("selected:", sc.selected)
print("IV:", sc.iv.round(3).to_dict())
print("test:", out["test"])
print("LC sub-grade test:", out["lc_subgrade_test"])

# Figure 3: information value
fig, ax = plt.subplots(figsize=(7, 4.5))
iv = sc.iv.sort_values()
colors = ["#2166ac" if v >= 0.02 else "#bbbbbb" for v in iv]
ax.barh(iv.index, iv.values, color=colors)
ax.axvline(0.02, color="k", ls="--", lw=1)
ax.set_xlabel("information value (training set); grey = dropped (IV < 0.02)")
ax.set_title("Information value of each application feature")
savefig(fig, "fig03_information_value")

# Figure 4: WoE bins of the four strongest features
fig, axes = plt.subplots(1, 4, figsize=(13, 3.4))
for ax, f in zip(axes, sc.iv.index[:4]):
    t = sc.binnings[f].table
    ax.bar(range(len(t)), t["default_rate"] * 100, color="#9bb7d4")
    ax.set_xticks(range(len(t)))
    ax.set_xticklabels([str(i)[:14] for i in t.index], rotation=70, fontsize=6)
    ax.set_ylabel("default rate (%)")
    ax.set_title(f"{f} (IV {sc.iv[f]:.3f})")
fig.suptitle("Monotone bins of the four features with the highest information value")
savefig(fig, "fig04_woe_bins")

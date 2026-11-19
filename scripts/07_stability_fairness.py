"""Stability (PSI) and performance by sub-group for the champion model.

- PSI of the score: training vs. validation, test, and every issue quarter 2016Q1-2018Q4
  (later applications whose outcome is not yet known: the PSI needs no outcome).
- PSI of each input feature, training vs. test.
- Sub-groups on the test set: AUC, observed default rate and calibrated PD by home ownership,
  region, income quintile, employment length and income verification. Lending Club holds no
  protected attribute (age, sex, ethnicity), so this is a stability-by-segment check, not a
  full fair-lending test.

Output: results/stability.json, figures 12-13.
"""

import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src import config  # noqa: E402
from src.data import load_loans, modelling_sample, time_split  # noqa: E402
from src.models import predict_named  # noqa: E402
from src.plotting import savefig, style  # noqa: E402
from src.stability import performance_by_group, psi, psi_categorical  # noqa: E402

style()
loans = load_loans()
train, valid, test = time_split(modelling_sample(loans))
champion = json.load(open(config.RESULTS / "calibration.json"))["champion"]
fitted = pickle.load(open(config.PROCESSED / "models.pkl", "rb"))
scorecard = pickle.load(open(config.PROCESSED / "scorecard.pkl", "rb"))
final = pd.read_parquet(config.PROCESSED / "pd_final.parquet")
X = config.FEATURES

s_train = final.loc[final["split"] == "train", "pd_raw"].to_numpy()
s_valid = final.loc[final["split"] == "valid", "pd_raw"].to_numpy()
s_test = final.loc[final["split"] == "test", "pd_raw"].to_numpy()

later = loans[loans["issue_d"] > config.TEST_END]
quarters = {}
for q, d in later.groupby(later["issue_d"].dt.to_period("Q")):
    s = predict_named(champion, fitted, scorecard, d[X])
    quarters[str(q)] = {"n": int(len(d)), "psi": psi(s_train, s), "mean_raw_pd": float(s.mean())}
    print(q, quarters[str(q)], flush=True)

feature_psi = {}
for c in X:
    if c in config.CATEGORICAL_FEATURES:
        feature_psi[c] = psi_categorical(train[c], test[c])
    else:
        a, b = train[c].dropna(), test[c].dropna()
        feature_psi[c] = psi(a, b)
feature_psi = dict(sorted(feature_psi.items(), key=lambda kv: -kv[1]))

df = test.copy()
df["pd"] = final.loc[final["split"] == "test", "pd"].to_numpy()
df["income_quintile"] = pd.qcut(df["annual_inc"], 5, labels=["Q1 (lowest)", "Q2", "Q3", "Q4", "Q5 (highest)"])
df["employment"] = pd.cut(df["emp_years"].fillna(-1), [-2, -0.5, 1.5, 4.5, 9.5, 11],
                          labels=["missing", "0-1 y", "2-4 y", "5-9 y", "10+ y"])
groups = {}
for g in ["home_ownership", "region", "income_quintile", "employment", "verification_status"]:
    t = performance_by_group(df, g)
    groups[g] = t.round(4).to_dict(orient="records")

out = {"champion": champion,
       "psi_score": {"train_vs_valid": psi(s_train, s_valid), "train_vs_test": psi(s_train, s_test)},
       "psi_by_quarter": quarters, "psi_features_train_vs_test": feature_psi, "groups": groups}
json.dump(out, open(config.RESULTS / "stability.json", "w"), indent=2, default=float)
print(json.dumps({k: out[k] for k in ["psi_score", "psi_features_train_vs_test"]}, indent=1))

# Figure 12: PSI by quarter
fig, ax = plt.subplots(figsize=(9, 3.6))
labels = ["valid\n2015H1", "test\n2015H2"] + list(quarters)
vals = [out["psi_score"]["train_vs_valid"], out["psi_score"]["train_vs_test"]] + [v["psi"] for v in quarters.values()]
ax.bar(range(len(vals)), vals, color=["#2166ac"] * 2 + ["#9bb7d4"] * len(quarters))
ax.axhline(0.10, color="orange", ls="--", lw=1, label="0.10: monitor")
ax.axhline(0.25, color="red", ls="--", lw=1, label="0.25: significant shift")
ax.set_xticks(range(len(vals)))
ax.set_xticklabels(labels, rotation=60, fontsize=7)
ax.set_ylabel("PSI of the score vs. training")
ax.set_title(f"Population stability of the {champion} score")
ax.legend(fontsize=8)
savefig(fig, "fig12_psi")

# Figure 13: sub-groups
fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
for ax, g in zip(axes, ["home_ownership", "income_quintile", "region"]):
    t = pd.DataFrame(groups[g])
    x = np.arange(len(t))
    ax.bar(x - 0.2, t["observed_dr"] * 100, 0.4, label="observed default rate", color="#9bb7d4")
    ax.bar(x + 0.2, t["predicted_pd"] * 100, 0.4, label="mean calibrated PD", color="#b2182b")
    for xi, a in zip(x, t["auc"]):
        ax.text(xi, 1, f"AUC\n{a:.3f}", ha="center", fontsize=7, color="white")
    ax.set_xticks(x)
    ax.set_xticklabels(t["group"], rotation=30, fontsize=7)
    ax.set_title(g)
    ax.set_ylabel("%")
axes[0].legend(fontsize=7)
fig.suptitle("Test set: observed vs. predicted default rate and AUC by segment")
savefig(fig, "fig13_subgroups")

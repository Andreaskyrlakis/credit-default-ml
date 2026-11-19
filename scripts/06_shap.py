"""Explain the LightGBM model with SHAP (TreeExplainer, exact for trees, in log-odds units).

Global: mean |SHAP| per feature and the beeswarm on 5,000 test loans.
Local: one test client, the waterfall of contributions from the average log-odds to its PD.

Output: results/shap.json, figures 9-11.
"""

import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402

from src import config  # noqa: E402
from src.data import load_loans, modelling_sample, time_split  # noqa: E402
from src.plotting import savefig, style  # noqa: E402

style()
_, _, test = time_split(modelling_sample(load_loans()))
pipe = pickle.load(open(config.PROCESSED / "models.pkl", "rb"))["lightgbm"]
prep, model = pipe.named_steps["prep"], pipe.named_steps["model"]

sample = test.sample(5000, random_state=config.SEED)
Xs = pd.DataFrame(prep.transform(sample[config.FEATURES]),
                  columns=[n.split("__", 1)[1] for n in prep.get_feature_names_out()],
                  index=sample.index)
explainer = shap.TreeExplainer(model)
sv = explainer(Xs)
if sv.values.ndim == 3:          # some versions return one slice per class
    sv = sv[:, :, 1]


def original(col):
    """Map a one-hot column back to its original feature."""
    for c in config.CATEGORICAL_FEATURES:
        if col.startswith(c + "_"):
            return c
    return col


abs_mean = pd.Series(np.abs(sv.values).mean(0), index=Xs.columns)
grouped = abs_mean.groupby(abs_mean.index.map(original)).sum().sort_values(ascending=False)

# check: SHAP values + base value = model log-odds
raw = model.predict(Xs, raw_score=True)
additivity_error = float(np.max(np.abs(sv.values.sum(1) + sv.base_values - raw)))

# local explanations: the riskiest defaulter in the sample, and the median client
pd_sample = model.predict_proba(Xs)[:, 1]
cand = np.where(sample[config.TARGET].to_numpy() == 1)[0]
i = int(cand[np.argmax(pd_sample[cand])])
j = int(np.argmin(np.abs(pd_sample - np.median(pd_sample))))
local = {}
for tag, k in [("high_risk_defaulter", i), ("median_client", j)]:
    contrib = pd.Series(sv.values[k], index=Xs.columns)
    top = contrib.reindex(contrib.abs().sort_values(ascending=False).index)[:8]
    local[tag] = {"pd": float(pd_sample[k]), "defaulted": int(sample[config.TARGET].iloc[k]),
                  "base_pd": float(1 / (1 + np.exp(-sv.base_values[k]))),
                  "top_contributions_logodds": top.round(4).to_dict(),
                  "features": {c: (None if pd.isna(sample[c].iloc[k]) else
                                   float(sample[c].iloc[k]) if c in config.NUMERIC_FEATURES
                                   else str(sample[c].iloc[k])) for c in config.FEATURES}}

json.dump({"mean_abs_shap_grouped": grouped.round(5).to_dict(),
           "additivity_max_error": additivity_error, "n_explained": len(sample), "local": local},
          open(config.RESULTS / "shap.json", "w"), indent=2)
print(grouped.round(4))
print("additivity error", additivity_error)
print(json.dumps(local, indent=1)[:2000])

fig, ax = plt.subplots(figsize=(6, 4.5))
g = grouped[:12][::-1]
ax.barh(g.index, g.values, color="#2166ac")
ax.set_xlabel("mean |SHAP value| (log-odds of default)")
ax.set_title("Global importance, LightGBM (5,000 test loans)")
savefig(fig, "fig09_shap_importance")

plt.figure()
shap.plots.beeswarm(sv, max_display=12, show=False)
fig = plt.gcf()
fig.set_size_inches(8, 5)
plt.title("SHAP value per loan (red = high feature value)")
savefig(fig, "fig10_shap_beeswarm")

plt.figure()
shap.plots.waterfall(sv[i], max_display=10, show=False)
fig = plt.gcf()
fig.set_size_inches(8, 5)
plt.title(f"One client: predicted PD {pd_sample[i]:.1%} (contributions in log-odds)")
savefig(fig, "fig11_shap_waterfall")

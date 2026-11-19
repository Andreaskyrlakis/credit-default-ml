"""Calibrate the champion model (best validation AUC) on the validation set.

Platt scaling and isotonic regression are fitted on the validation loans (January-June 2015)
and judged on the test loans. The method kept for IFRS 9 is chosen on the validation set by
5-fold cross-fitting (each fold calibrated on the other four), never on the test set.

Output: data/processed/pd_final.parquet, results/calibration.json, figure 8.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.model_selection import KFold  # noqa: E402

from src import config  # noqa: E402
from src.calibration import IsotonicCalibrator, PlattCalibrator  # noqa: E402
from src.data import load_loans, modelling_sample, time_split  # noqa: E402
from src.metrics import all_metrics, calibration_bins, expected_calibration_error  # noqa: E402
from src.plotting import savefig, style  # noqa: E402

style()
models = json.load(open(config.RESULTS / "models.json"))["results"]
champion = max(models, key=lambda k: models[k]["valid_auc"])
print("champion (validation AUC):", champion)

train, valid, test = time_split(modelling_sample(load_loans()))
preds = pd.read_parquet(config.PROCESSED / "predictions.parquet")
y = config.TARGET
pv = preds.loc[preds["split"] == "valid", champion].to_numpy()
pt = preds.loc[preds["split"] == "test", champion].to_numpy()
ptr = preds.loc[preds["split"] == "train", champion].to_numpy()
yv, yt = valid[y].to_numpy(), test[y].to_numpy()

# choose the method by cross-fitting on validation
cross = {"raw": [], "platt": [], "isotonic": []}
for tr_idx, te_idx in KFold(5, shuffle=True, random_state=config.SEED).split(pv):
    cross["raw"].append(expected_calibration_error(yv[te_idx], pv[te_idx]))
    for name, cls in [("platt", PlattCalibrator), ("isotonic", IsotonicCalibrator)]:
        c = cls().fit(pv[tr_idx], yv[tr_idx])
        cross[name].append(expected_calibration_error(yv[te_idx], c.transform(pv[te_idx])))
cross = {k: float(np.mean(v)) for k, v in cross.items()}
chosen = min(["platt", "isotonic"], key=lambda k: cross[k])

platt = PlattCalibrator().fit(pv, yv)
iso = IsotonicCalibrator().fit(pv, yv)
test_versions = {"raw": pt, "platt": platt.transform(pt), "isotonic": iso.transform(pt)}
res = {"champion": champion, "chosen_method": chosen, "cv_ece_on_valid": cross,
       "platt_slope_intercept": platt.slope_intercept,
       "mean_pd_train_raw": float(ptr.mean()), "default_rate_train": float(train[y].mean()),
       "default_rate_valid": float(yv.mean()), "default_rate_test": float(yt.mean()),
       "test": {}}
for k, p in test_versions.items():
    m = all_metrics(yt, p)
    m["ece"] = expected_calibration_error(yt, p)
    m["mean_pd"] = float(p.mean())
    res["test"][k] = m
json.dump(res, open(config.RESULTS / "calibration.json", "w"), indent=2)
print(json.dumps(res, indent=2))

final = iso if chosen == "isotonic" else platt
out = preds[["id", "split"]].copy()
out["default"] = pd.concat([train[y], valid[y], test[y]]).to_numpy()
out["pd_raw"] = preds[champion].to_numpy()
out["pd"] = final.transform(out["pd_raw"].to_numpy())
out.to_parquet(config.PROCESSED / "pd_final.parquet")

# Figure 8: reliability diagram on the test set
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
ax = axes[0]
for k, c in [("raw", "#999999"), ("platt", "#2166ac"), ("isotonic", "#b2182b")]:
    pred, obs = calibration_bins(yt, test_versions[k], 10)
    ece = res["test"][k]["ece"] * 100
    ax.plot(pred * 100, obs * 100, marker="o", color=c, label=f"{k} (ECE {ece:.2f} pp)")
lim = max(ax.get_xlim()[1], ax.get_ylim()[1])
ax.plot([0, lim], [0, lim], "k--", lw=0.8)
ax.set_xlabel("mean predicted PD in decile (%)")
ax.set_ylabel("observed default rate (%)")
ax.set_title(f"Reliability, test set ({champion})")
ax.legend(fontsize=8)
ax = axes[1]
ax.hist(test_versions["raw"], bins=60, alpha=0.6, label="raw", color="#999999")
ax.hist(test_versions[chosen], bins=60, alpha=0.6, label=chosen, color="#b2182b")
ax.set_xlabel("predicted lifetime PD")
ax.set_ylabel("loans")
ax.set_title("Distribution of predicted PDs, test set")
ax.legend(fontsize=8)
savefig(fig, "fig08_calibration")

"""Tune and compare the models.

1. Hyperparameters: randomised search, 5-fold shuffled stratified cross-validation, on a random
   sample of 60,000 training loans (CPU and memory budget); score = ROC AUC.
2. Each tuned model is refitted on the full training set (2007-2014).
3. Test set (July-December 2015): AUC, Gini, KS, Brier, PR-AUC with 95% bootstrap intervals.
4. Learning curves: validation AUC against the number of training loans.

Output: data/processed/predictions.parquet, results/models.json, figures 5-7.
"""

import json
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score, roc_curve  # noqa: E402
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold  # noqa: E402

from src import config  # noqa: E402
from src.data import load_loans, modelling_sample, time_split  # noqa: E402
from src.metrics import bootstrap_metrics, paired_auc_difference  # noqa: E402
from src.models import N_ITER, SEARCH_SPACES, make_model  # noqa: E402
from src.plotting import savefig, style  # noqa: E402

style()
N_BOOT = 300
train, valid, test = time_split(modelling_sample(load_loans()))
X, y = config.FEATURES, config.TARGET
tune = train.sample(60_000, random_state=config.SEED)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.SEED)

LABELS = {"scorecard": "WoE scorecard", "logistic": "Logistic (raw)", "tree": "Decision tree",
          "random_forest": "Random forest", "hist_gb": "Hist. gradient boosting",
          "lightgbm": "LightGBM", "mlp": "MLP (neural net)", "vote": "Soft vote (LR+LGBM+MLP)"}

fitted, tuning = {}, {}
for name in ["logistic", "tree", "random_forest", "hist_gb", "lightgbm", "mlp"]:
    t0 = time.time()
    search = RandomizedSearchCV(make_model(name), SEARCH_SPACES[name], n_iter=N_ITER[name],
                                scoring="roc_auc", cv=cv, random_state=config.SEED, n_jobs=1,
                                refit=False)
    search.fit(tune[X], tune[y])
    best = {k.replace("model__", ""): v for k, v in search.best_params_.items()}
    best = {k: (v.item() if hasattr(v, "item") else v) for k, v in best.items()}
    model = make_model(name, **best).fit(train[X], train[y])
    fitted[name] = model
    tuning[name] = {"best_params": {k: (list(v) if isinstance(v, tuple) else v) for k, v in best.items()},
                    "cv_auc_mean": float(search.best_score_),
                    "cv_auc_std": float(search.cv_results_["std_test_score"][search.best_index_]),
                    "n_candidates": N_ITER[name], "seconds": round(time.time() - t0, 1)}
    print(name, tuning[name], flush=True)

scorecard = pickle.load(open(config.PROCESSED / "scorecard.pkl", "rb"))
pickle.dump(fitted, open(config.PROCESSED / "models.pkl", "wb"))


def predict_all(df):
    out = {"scorecard": scorecard.predict_proba(df[X])[:, 1]}
    for name, m in fitted.items():
        out[name] = m.predict_proba(df[X])[:, 1]
    out["vote"] = (out["logistic"] + out["lightgbm"] + out["mlp"]) / 3
    return out


preds = []
for split, df in [("train", train), ("valid", valid), ("test", test)]:
    p = pd.DataFrame(predict_all(df), index=df.index)
    p.insert(0, "split", split)
    p.insert(0, "id", df["id"].astype(str).values)
    preds.append(p)
preds = pd.concat(preds)
preds.to_parquet(config.PROCESSED / "predictions.parquet")

yt = test[y].to_numpy()
pt = preds[preds["split"] == "test"]
pv = preds[preds["split"] == "valid"]
results = {}
for name in LABELS:
    results[name] = {"test": bootstrap_metrics(yt, pt[name].to_numpy(), n_boot=N_BOOT, seed=1),
                     "valid_auc": float(roc_auc_score(valid[y], pv[name])),
                     "train_auc": float(roc_auc_score(train[y], preds[preds["split"] == "train"][name]))}
    print(name, {k: round(v["value"], 4) for k, v in results[name]["test"].items()}, flush=True)
diff = paired_auc_difference(yt, pt["lightgbm"].to_numpy(), pt["scorecard"].to_numpy(),
                             n_boot=N_BOOT, seed=2)
diff_vote = paired_auc_difference(yt, pt["vote"].to_numpy(), pt["lightgbm"].to_numpy(),
                                  n_boot=N_BOOT, seed=3)

# Learning curves (validation AUC) for the linear model and LightGBM
sizes = [2_000, 5_000, 10_000, 25_000, 50_000, 100_000, 200_000, len(train)]
lc = {}
for name in ["logistic", "lightgbm"]:
    params = tuning[name]["best_params"]
    params = {k: (tuple(v) if isinstance(v, list) else v) for k, v in params.items()}
    rows = []
    for n in sizes:
        sub = train.sample(n, random_state=config.SEED) if n < len(train) else train
        m = make_model(name, **params).fit(sub[X], sub[y])
        rows.append({"n": n, "train_auc": float(roc_auc_score(sub[y], m.predict_proba(sub[X])[:, 1])),
                     "valid_auc": float(roc_auc_score(valid[y], m.predict_proba(valid[X])[:, 1]))})
    lc[name] = rows
    print(name, "learning curve", rows[-1], flush=True)

json.dump({"tuning": tuning, "results": results, "lightgbm_minus_scorecard_auc": diff,
           "vote_minus_lightgbm_auc": diff_vote, "learning_curves": lc, "n_boot": N_BOOT,
           "tuning_sample": len(tune)},
          open(config.RESULTS / "models.json", "w"), indent=2)

# Figure 5: AUC with bootstrap intervals
order = sorted(LABELS, key=lambda k: results[k]["test"]["auc"]["value"])
fig, ax = plt.subplots(figsize=(7, 4))
vals = [results[k]["test"]["auc"]["value"] for k in order]
lo = [vals[i] - results[k]["test"]["auc"]["low"] for i, k in enumerate(order)]
hi = [results[k]["test"]["auc"]["high"] - vals[i] for i, k in enumerate(order)]
ax.errorbar(vals, range(len(order)), xerr=[lo, hi], fmt="o", capsize=3, color="#2166ac")
ax.set_yticks(range(len(order)))
ax.set_yticklabels([LABELS[k] for k in order])
ax.set_xlabel("test ROC AUC (95% bootstrap interval)")
ax.set_title(f"Model comparison on the test set ({len(test):,} loans, July-December 2015)")
savefig(fig, "fig05_model_auc")

# Figure 6: ROC curves
fig, ax = plt.subplots(figsize=(5, 5))
for k in ["scorecard", "logistic", "tree", "random_forest", "lightgbm", "mlp"]:
    fpr, tpr, _ = roc_curve(yt, pt[k])
    ax.plot(fpr, tpr, lw=1.2, label=f"{LABELS[k]} ({results[k]['test']['auc']['value']:.3f})")
ax.plot([0, 1], [0, 1], "k--", lw=0.8)
ax.set_xlabel("false positive rate (good loans flagged)")
ax.set_ylabel("true positive rate (defaults caught)")
ax.legend(fontsize=7)
ax.set_title("ROC curves, test set")
savefig(fig, "fig06_roc")

# Figure 7: learning curves
fig, ax = plt.subplots(figsize=(6, 4))
for name, c in [("logistic", "#b2182b"), ("lightgbm", "#2166ac")]:
    n = [r["n"] for r in lc[name]]
    ax.plot(n, [r["valid_auc"] for r in lc[name]], marker="o", color=c, label=f"{LABELS[name]}: validation")
    ax.plot(n, [r["train_auc"] for r in lc[name]], ls="--", color=c, label=f"{LABELS[name]}: training")
ax.set_xscale("log")
ax.set_xlabel("number of training loans")
ax.set_ylabel("ROC AUC")
ax.legend(fontsize=7)
ax.set_title("Learning curves")
savefig(fig, "fig07_learning_curves")

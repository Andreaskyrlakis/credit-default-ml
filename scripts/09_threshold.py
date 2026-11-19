"""Choose the approval cut-off on the calibrated PD.

Costs come from the data (training loans), as a share of the amount lent:
- c_fp, refusing a good client = average realised profit of a fully paid loan
  (total repaid - amount lent) / amount lent;
- c_fn, accepting a defaulter = average realised loss of a charged-off loan.
Bayes cut-off PD* = c_fp / (c_fp + c_fn). The profit-maximising cut-off is also searched on the
validation set (realised profit per loan), then both are applied unchanged to the test set.
Funding cost, operating cost and time value of money are ignored (see the report).

Output: results/threshold.json, figure 17.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src import config  # noqa: E402
from src.data import load_loans, modelling_sample, time_split  # noqa: E402
from src.plotting import savefig, style  # noqa: E402
from src.threshold import bayes_cutoff, best_cutoff, cost_curve, profit_curve  # noqa: E402

style()
train, valid, test = time_split(modelling_sample(load_loans()))
final = pd.read_parquet(config.PROCESSED / "pd_final.parquet")
y = config.TARGET

ret = (train["total_pymnt"] - train["funded_amnt"]) / train["funded_amnt"]
c_fp = float(ret[train[y] == 0].mean())
c_fn = float(-ret[train[y] == 1].mean())
cut_bayes = bayes_cutoff(c_fp, c_fn)

pv = final.loc[final["split"] == "valid", "pd"].to_numpy()
pt = final.loc[final["split"] == "test", "pd"].to_numpy()
profit_v = (valid["total_pymnt"] - valid["funded_amnt"]).to_numpy()
profit_t = (test["total_pymnt"] - test["funded_amnt"]).to_numpy()
grid = np.round(np.linspace(0.02, 0.60, 117), 4)
_, tot_v, appr_v = profit_curve(pv, profit_v, grid)
cut_profit = best_cutoff(grid, tot_v)
_, cost_v = cost_curve(valid[y].to_numpy(), pv, c_fp, c_fn, grid)
cut_cost = best_cutoff(grid, cost_v, maximise=False)


def evaluate(cut, p, df, profit):
    acc = p < cut
    yy = df[y].to_numpy()
    return {"cutoff": float(cut), "approval_rate": float(acc.mean()),
            "default_rate_approved": float(yy[acc].mean()),
            "default_rate_refused": float(yy[~acc].mean()) if (~acc).any() else None,
            "good_refused_share": float(((~acc) & (yy == 0)).sum() / (yy == 0).sum()),
            "bad_caught_share": float(((~acc) & (yy == 1)).sum() / (yy == 1).sum()),
            "profit_usd": float(profit[acc].sum()),
            "profit_per_dollar_lent": float(profit[acc].sum() / df["funded_amnt"].to_numpy()[acc].sum())}


test_eval = {"approve_all": evaluate(1.01, pt, test, profit_t),
             "bayes_cutoff": evaluate(cut_bayes, pt, test, profit_t),
             "cost_min_on_valid": evaluate(cut_cost, pt, test, profit_t),
             "profit_max_on_valid": evaluate(cut_profit, pt, test, profit_t)}
out = {"c_fp_refuse_good": c_fp, "c_fn_accept_bad": c_fn, "cost_ratio_fn_over_fp": c_fn / c_fp,
       "bayes_cutoff": cut_bayes, "cost_min_cutoff_valid": cut_cost,
       "profit_max_cutoff_valid": cut_profit, "test": test_eval}
json.dump(out, open(config.RESULTS / "threshold.json", "w"), indent=2)
print(json.dumps(out, indent=2))

_, tot_t, appr_t = profit_curve(pt, profit_t, grid)
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
axes[0].plot(grid * 100, tot_v / 1e6, label="validation (choice)", color="#2166ac")
axes[0].plot(grid * 100, tot_t / 1e6, label="test", color="#b2182b")
axes[0].axvline(cut_profit * 100, color="#2166ac", ls="--", lw=1)
axes[0].axvline(cut_bayes * 100, color="k", ls=":", lw=1, label=f"Bayes cut-off {cut_bayes:.1%}")
axes[0].set_xlabel("approve if calibrated PD < cut-off (%)")
axes[0].set_ylabel("realised profit of approved loans (USD m)")
axes[0].legend(fontsize=7)
axes[0].set_title("Realised profit against the cut-off")
axes[1].plot(appr_t * 100, [test[y].to_numpy()[pt < c].mean() * 100 if (pt < c).any() else np.nan
                            for c in grid], color="#b2182b")
axes[1].set_xlabel("approval rate (%)")
axes[1].set_ylabel("default rate of approved loans (%)")
axes[1].set_title("Approval rate vs. default rate (test)")
savefig(fig, "fig17_threshold")

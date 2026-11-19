"""Clean the raw file into one table of 36-month loans and describe the modelling sample.

Output: data/processed/loans_36m.parquet, results/data_summary.json, figures 1-2.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from src import config  # noqa: E402
from src.data import build_loan_table, modelling_sample, time_split  # noqa: E402
from src.plotting import savefig, style  # noqa: E402

style()
config.PROCESSED.mkdir(parents=True, exist_ok=True)
config.RESULTS.mkdir(exist_ok=True)

loans = build_loan_table()
loans.to_parquet(config.LOANS_FILE, index=False)
print("36-month individual loans:", len(loans))

sample = modelling_sample(loans)
train, valid, test = time_split(sample)
issued_to_2015 = loans[loans["issue_d"] <= config.TEST_END]
unresolved = issued_to_2015[config.TARGET].isna().sum()

summary = {
    "raw_rows_36m_individual": int(len(loans)),
    "issued_2007_2015": int(len(issued_to_2015)),
    "unresolved_2007_2015_dropped": int(unresolved),
    "modelling_sample": int(len(sample)),
    "default_rate": float(sample[config.TARGET].mean()),
    "splits": {name: {"n": int(len(d)), "defaults": int(d[config.TARGET].sum()),
                      "default_rate": float(d[config.TARGET].mean()),
                      "first_issue": str(d["issue_d"].min().date()),
                      "last_issue": str(d["issue_d"].max().date())}
               for name, d in [("train", train), ("valid", valid), ("test", test)]},
    "missing_share": {c: float(sample[c].isna().mean()) for c in config.NUMERIC_FEATURES
                      if sample[c].isna().any()},
    "default_rate_by_grade": sample.groupby("grade", observed=True)[config.TARGET].mean().round(4).to_dict(),
}
json.dump(summary, open(config.RESULTS / "data_summary.json", "w"), indent=2)
print(json.dumps(summary["splits"], indent=2))

# Figure 1: volume and default rate by issue quarter, with the split
q = sample.groupby(sample["issue_d"].dt.to_period("Q"))[config.TARGET].agg(["size", "mean"])
fig, ax = plt.subplots(figsize=(9, 4))
x = q.index.to_timestamp()
ax.bar(x, q["size"] / 1000, width=80, color="#9bb7d4", label="loans issued (thousands)")
ax.set_ylabel("loans issued per quarter (thousands)")
ax2 = ax.twinx()
ax2.plot(x, q["mean"] * 100, color="#b2182b", marker="o", ms=3, label="default rate (%)")
ax2.set_ylabel("default rate (%)")
top = ax.get_ylim()[1]
for d, lab, y_pos in [(config.TRAIN_END, "train | valid ", 0.95), (config.VALID_END, "valid | test ", 0.85)]:
    ax.axvline(pd.Timestamp(d), color="k", ls="--", lw=1)
    ax.text(pd.Timestamp(d), top * y_pos, lab, fontsize=8, va="top", ha="right")
ax.set_title("36-month Lending Club loans: volume and realised default rate by issue quarter")
fig.legend(loc="upper left", bbox_to_anchor=(0.08, 0.88), fontsize=8)
savefig(fig, "fig01_volume_default_rate")

# Figure 2: default rate along four application features (training sample)
fig, axes = plt.subplots(1, 4, figsize=(12, 3.2))
for ax, col in zip(axes, ["fico", "dti", "loan_to_income", "inq_last_6mths"]):
    b = pd.qcut(train[col], 10, duplicates="drop")
    r = train.groupby(b, observed=True)[config.TARGET].mean() * 100
    mids = [iv.mid for iv in r.index]
    ax.plot(mids, r.values, marker="o")
    ax.set_title(col)
    ax.set_ylabel("default rate (%)")
fig.suptitle("Default rate by decile of four application features (training set)")
savefig(fig, "fig02_default_by_feature")

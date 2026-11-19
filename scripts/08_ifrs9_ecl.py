"""IFRS 9 expected credit loss on the test portfolio at 30 June 2016.

Portfolio: the 36-month loans issued July-December 2015 (the test set), aged 6-11 months at the
reporting date. Inputs:
- lifetime PD at origination: champion model, calibrated (script 05);
- timing curve of defaults and LGD: training loans only (issued up to 2014);
- unemployment link: FRED charge-off rate on bank consumer loans vs. unemployment rate;
- three scenarios with assumed weights and unemployment shocks (stated below).
Back-test: the realised principal loss after the reporting date, known from the data.

Output: results/ifrs9.json, figures 14-16.
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
from src.ifrs9 import (TERM, balance_schedule, default_month, ecl_loan_level,  # noqa: E402
                       empirical_lgd, fit_macro_link, run_ecl, status_at,
                       timing_curve, window_pd)
from src.plotting import savefig, style  # noqa: E402

style()
# Scenario design (assumptions): change in the US unemployment rate vs. the base path, in pp
SCENARIOS = {"upside": (0.15, -1.0), "base": (0.60, 0.0), "downside": (0.25, 3.0)}

train, valid, test = time_split(modelling_sample(load_loans()))
final = pd.read_parquet(config.PROCESSED / "pd_final.parquet")

# 1. timing curve and LGD from training defaults
bad_train = train[train[config.TARGET] == 1]
months = default_month(bad_train["issue_d"], bad_train["last_pymnt_d"])
F = timing_curve(months, TERM)
lgd_loans = empirical_lgd(bad_train)
ead_obs = (bad_train["funded_amnt"] - bad_train["total_rec_prncp"]).clip(lower=1)
lgd = float(np.average(lgd_loans, weights=ead_obs))

# 2. macro link
u = pd.read_csv(config.RAW / "fred_unrate.csv", parse_dates=["observation_date"], index_col=0)["UNRATE"]
co = pd.read_csv(config.RAW / "fred_corcacbs.csv", parse_dates=["observation_date"], index_col=0)["CORCACBS"]
macro = fit_macro_link(u, co)
beta = macro["beta_unemployment"]
u_at_R = float(u.loc[:config.REPORTING_DATE].iloc[-1])

# 3. book at the reporting date
st = status_at(test, config.REPORTING_DATE)
book = test.join(st)
book["pd"] = final.loc[final["split"] == "test", "pd"].to_numpy()
open_book = book[book["open"]].copy()
res = run_ecl(open_book, F, lgd, beta, SCENARIOS)
open_book["stage"] = res["stage"]
open_book["ead"] = res["ead"]
open_book["ecl"] = res["ecl"]
for name, r in res["scenarios"].items():
    open_book[f"ecl_{name}"] = r["ecl"]

age = open_book["age"].to_numpy()
p0 = open_book["pd"].to_numpy()
pd12 = window_pd(p0, F, age, 12)
pdlife = window_pd(p0, F, age, TERM)

by_stage = open_book.groupby("stage").agg(n=("id", "size"), ead=("ead", "sum"), ecl=("ecl", "sum"))
by_stage["coverage"] = by_stage["ecl"] / by_stage["ead"]
scen_tot = {n: {"weight": w, "delta_unemployment": du, "unemployment": u_at_R + du,
                "ecl": float(open_book[f"ecl_{n}"].sum())} for n, (w, du) in SCENARIOS.items()}

# 4. back-test: principal actually lost on loans performing at R that defaulted later
perf = open_book[open_book["stage"] < 3]
later = perf[perf["defaults_later"]]
realised_loss = float(((later["funded_amnt"] - later["total_rec_prncp"])
                       - (later["recoveries"] - later["collection_recovery_fee"]).clip(lower=0)).sum())
bal = balance_schedule(perf["funded_amnt"], perf["int_rate"])
undisc_life_base = float(ecl_loan_level(perf["pd"].to_numpy(), F, perf["age"].to_numpy(), bal,
                                        np.zeros(len(perf)), lgd, TERM).sum())
realised_default_rate = float(perf["defaults_later"].mean())
predicted_default_rate = float(window_pd(perf["pd"].to_numpy(), F, perf["age"].to_numpy(), TERM).mean())

out = {
    "reporting_date": config.REPORTING_DATE, "lgd_ead_weighted": lgd,
    "lgd_mean": float(lgd_loans.mean()), "lgd_median": float(lgd_loans.median()),
    "n_training_defaults": int(len(bad_train)),
    "timing_share_first_12m": float(F[12]), "timing_share_first_24m": float(F[24]),
    "macro_link": macro, "unemployment_at_reporting_date": u_at_R,
    "portfolio": {"loans_in_test": int(len(test)), "open_at_R": int(len(open_book)),
                  "repaid_before_R": int((~book["open"] & (book["dpd"] <= 150)).sum()),
                  "charged_off_before_R": int((book["dpd"] > 150).sum()),
                  "ead_total": float(open_book["ead"].sum())},
    "pd_12m_mean": float(pd12.mean()), "pd_lifetime_remaining_mean": float(pdlife.mean()),
    "stages": {int(k): {kk: float(vv) for kk, vv in v.items()} for k, v in by_stage.to_dict(orient="index").items()},
    "stage2_by_dpd": int(((open_book["stage"] == 2) & (open_book["dpd"] > 30)).sum()),
    "stage2_by_sicr_pd_test": int(((open_book["stage"] == 2) & (open_book["dpd"] <= 30)).sum()),
    "ecl_weighted": float(open_book["ecl"].sum()),
    "coverage_weighted": float(open_book["ecl"].sum() / open_book["ead"].sum()),
    "scenarios": scen_tot,
    "ecl_all_stage1_12m_base": float(res["scenarios"]["base"]["ecl_12m"].sum()),
    "ecl_all_lifetime_base": float(res["scenarios"]["base"]["ecl_life"].sum()),
    "backtest": {"performing_loans": int(len(perf)), "defaulted_after_R": int(len(later)),
                 "realised_default_rate": realised_default_rate,
                 "predicted_remaining_lifetime_pd": predicted_default_rate,
                 "realised_principal_loss": realised_loss,
                 "predicted_lifetime_loss_undiscounted_base": undisc_life_base},
}
json.dump(out, open(config.RESULTS / "ifrs9.json", "w"), indent=2)
print(json.dumps(out, indent=2))

# Figure 14: timing curve and LGD distribution
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
marg = np.diff(F)
axes[0].bar(np.arange(1, TERM + 1), marg * 100, color="#9bb7d4", label="share of defaults in month")
ax2 = axes[0].twinx()
ax2.plot(np.arange(TERM + 1), F * 100, color="#b2182b", label="cumulative")
axes[0].set_xlabel("loan month of the first missed payment")
axes[0].set_ylabel("% of lifetime defaults in the month")
ax2.set_ylabel("cumulative %")
axes[0].set_title(f"Default timing ({len(bad_train):,} training defaults)")
axes[1].hist(lgd_loans, bins=50, color="#2166ac")
axes[1].axvline(lgd, color="k", ls="--", label=f"EAD-weighted LGD {lgd:.1%}")
axes[1].set_xlabel("LGD = 1 - net recoveries / principal outstanding")
axes[1].set_ylabel("charged-off loans")
axes[1].legend(fontsize=8)
axes[1].set_title("Realised LGD of charged-off training loans")
savefig(fig, "fig14_timing_lgd")

# Figure 15: macro link
q_u = u.resample("QS").mean()
dfm = pd.concat([q_u.rename("u"), co.rename("co")], axis=1).dropna()
fig, ax = plt.subplots(figsize=(5.5, 4))
ax.scatter(dfm["u"], dfm["co"], s=10, color="#2166ac", alpha=0.7)
grid = np.linspace(dfm["u"].min(), dfm["u"].max(), 50)
ax.plot(grid, 100 / (1 + np.exp(-(macro["intercept"] + beta * grid))), color="#b2182b",
        label=f"logit fit, beta = {beta:.3f} per pp, R2 = {macro['r2']:.2f}")
ax.set_xlabel("US unemployment rate (%), quarterly mean")
ax.set_ylabel("charge-off rate, bank consumer loans (%)")
ax.set_title(f"Macro link (FRED, {macro['start'][:4]}-{macro['end'][:4]})")
ax.legend(fontsize=7)
savefig(fig, "fig15_macro_link")

# Figure 16: ECL by stage and scenario
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
st_tab = by_stage
axes[0].bar([f"Stage {i}\n{int(st_tab.loc[i, 'n']):,} loans" for i in st_tab.index],
            st_tab["coverage"] * 100, color=["#4daf4a", "#ff7f00", "#e41a1c"][:len(st_tab)])
for i, v in enumerate(st_tab["coverage"] * 100):
    axes[0].text(i, v, f"{v:.1f}%", ha="center", va="bottom")
axes[0].set_ylabel("ECL / EAD (%)")
axes[0].set_title("Coverage ratio by stage (probability-weighted)")
names = list(scen_tot)
vals = [scen_tot[n]["ecl"] / 1e6 for n in names] + [out["ecl_weighted"] / 1e6]
axes[1].bar([f"{n}\n(w={scen_tot[n]['weight']:.0%})" for n in names] + ["weighted"], vals,
            color=["#4daf4a", "#999999", "#e41a1c", "#2166ac"])
for i, v in enumerate(vals):
    axes[1].text(i, v, f"{v:.1f}", ha="center", va="bottom")
axes[1].set_ylabel("ECL (USD million)")
axes[1].set_title(f"ECL by macro scenario, book of {out['portfolio']['open_at_R']:,} loans")
savefig(fig, "fig16_ecl")

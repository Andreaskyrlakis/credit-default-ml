"""From a lifetime PD at origination to IFRS 9 expected credit losses (ECL).

Building blocks, each documented in the report:
- timing curve F(m): share of lifetime defaults that happen by loan month m (empirical, training);
- PD over any window, conditional on the loan still performing at its current age;
- EAD: scheduled balance of the amortising loan (no prepayment);
- LGD: empirical, from recoveries on charged-off training loans;
- staging at a reporting date from days past due (reconstructed from the last payment date)
  and a relative significant-increase-in-credit-risk (SICR) test;
- ECL = sum over months of marginal PD x LGD x EAD x discount factor at the loan's rate,
  over 12 months (stage 1) or the remaining life (stage 2), and LGD x EAD for stage 3;
- forward-looking: PDs shifted in logit space with unemployment, three weighted scenarios.
"""

import numpy as np
import pandas as pd
from scipy.special import expit, logit

TERM = 36


def months_between(start: pd.Series, end) -> pd.Series:
    """Whole months from each date in `start` to the single date `end`."""
    start, end = pd.to_datetime(start), pd.Timestamp(end)
    return (end.year - start.dt.year) * 12 + (end.month - start.dt.month)


def default_month(issue_d: pd.Series, last_pymnt_d: pd.Series) -> pd.Series:
    """Loan month of the first missed payment: one month after the last payment received."""
    m = (last_pymnt_d.dt.year - issue_d.dt.year) * 12 + (last_pymnt_d.dt.month - issue_d.dt.month) + 1
    return m.fillna(1).clip(lower=1, upper=TERM).astype(int)


def timing_curve(months_to_default, horizon=TERM) -> np.ndarray:
    """F[m] for m = 0..horizon: share of defaults that occurred by month m (F[0]=0, F[horizon]=1)."""
    counts = np.bincount(np.asarray(months_to_default, int), minlength=horizon + 1)[:horizon + 1]
    counts[0] = 0
    return np.cumsum(counts) / counts.sum()


def window_pd(p_life, F, age, months):
    """PD over (age, age+months], given no default by `age` (loan month)."""
    p_life, age = np.asarray(p_life, float), np.asarray(age, int)
    end = np.minimum(age + months, len(F) - 1)
    return p_life * (F[end] - F[age]) / (1 - p_life * F[age])


def balance_schedule(principal, annual_rate, term=TERM) -> np.ndarray:
    """Outstanding balance after k = 0..term monthly payments of an annuity loan (rows = loans)."""
    principal = np.asarray(principal, float)[:, None]
    r = np.asarray(annual_rate, float)[:, None] / 12.0
    k = np.arange(term + 1)[None, :]
    growth_n = (1 + r) ** term
    return principal * (growth_n - (1 + r) ** k) / (growth_n - 1)


def empirical_lgd(df: pd.DataFrame) -> pd.Series:
    """LGD of each charged-off loan = 1 - net recoveries / principal outstanding at default."""
    ead = (df["funded_amnt"] - df["total_rec_prncp"]).clip(lower=1.0)
    net_rec = (df["recoveries"] - df["collection_recovery_fee"]).clip(lower=0.0)
    return (1 - net_rec / ead).clip(0, 1)


def status_at(df: pd.DataFrame, reporting_date: str) -> pd.DataFrame:
    """Age, days past due and open/closed status of each loan at the reporting date."""
    R = pd.Timestamp(reporting_date)
    out = pd.DataFrame(index=df.index)
    out["age"] = months_between(df["issue_d"], R).astype(int).clip(0, TERM)
    paid_off = (df["default"] == 0) & (df["last_pymnt_d"] < R - pd.offsets.MonthBegin(1))
    first_missed = df["last_pymnt_d"].fillna(df["issue_d"]) + pd.DateOffset(months=1)
    dpd = (R - first_missed).dt.days.clip(lower=0)
    out["dpd"] = np.where(df["default"] == 1, dpd, 0)
    out["open"] = ~paid_off & (out["age"] < TERM)
    # Lending Club charges loans off after about 150 days past due; such loans are no longer on book
    out.loc[out["dpd"] > 150, "open"] = False
    # does the loan default after the reporting date? (used for the back-test only)
    out["defaults_later"] = (df["default"] == 1) & (out["dpd"] == 0)
    return out


def ecl_loan_level(p_life, F, age, balance, rate, lgd, months):
    """Sum over the next `months` loan months of marginal PD x LGD x EAD x discount factor."""
    p_life, age = np.asarray(p_life, float), np.asarray(age, int)
    n, H = len(p_life), len(F) - 1
    surv = 1 - p_life * F[age]
    r = np.asarray(rate, float) / 12.0
    ecl = np.zeros(n)
    for k in range(1, H + 1):
        m = age + k
        inside = (m <= H) & (k <= months)
        if not inside.any():
            break
        mm = np.minimum(m, H)
        marginal = p_life * (F[mm] - F[mm - 1]) / surv
        ead = balance[np.arange(n), mm - 1]   # balance due before the missed payment
        ecl += np.where(inside, marginal * lgd * ead / (1 + r) ** k, 0.0)
    return ecl


def fit_macro_link(unrate: pd.Series, chargeoff: pd.Series) -> dict:
    """OLS of logit(bank consumer-loan charge-off rate) on the unemployment rate (quarterly)."""
    q_u = unrate.resample("QS").mean()
    df = pd.concat([q_u.rename("u"), (chargeoff / 100).rename("co")], axis=1).dropna()
    X = np.column_stack([np.ones(len(df)), df["u"]])
    y = logit(df["co"].to_numpy())
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    r2 = 1 - resid.var() / y.var()
    return {"intercept": float(beta[0]), "beta_unemployment": float(beta[1]), "r2": float(r2),
            "n_quarters": int(len(df)), "start": str(df.index[0].date()), "end": str(df.index[-1].date())}


def scenario_pd(p_life, beta, delta_unemployment):
    """Shift the lifetime PD in logit space by beta x change in unemployment (percentage points)."""
    return expit(logit(np.clip(p_life, 1e-6, 1 - 1e-6)) + beta * delta_unemployment)


def run_ecl(book: pd.DataFrame, F, lgd, beta, scenarios: dict, sicr_multiple=2.0) -> dict:
    """Stage every open loan and compute ECL under each scenario and probability-weighted.

    book needs: pd (lifetime PD at origination), age, dpd, funded_amnt, int_rate.
    scenarios: {name: (weight, change in unemployment in pp)}.
    """
    age = book["age"].to_numpy()
    bal = balance_schedule(book["funded_amnt"], book["int_rate"])
    remaining = TERM - age
    p0 = book["pd"].to_numpy()
    # SICR test on the annualised remaining-life PD, weighted over scenarios, vs. origination
    w_life = sum(w * window_pd(scenario_pd(p0, beta, du), F, age, TERM)
                 for w, du in scenarios.values())
    annual_now = w_life / np.maximum(remaining, 1) * 12
    annual_orig = p0 / TERM * 12
    stage = np.ones(len(book), int)
    stage[(book["dpd"].to_numpy() > 30) | (annual_now >= sicr_multiple * annual_orig)] = 2
    stage[book["dpd"].to_numpy() > 90] = 3
    ead_now = bal[np.arange(len(book)), np.minimum(age, TERM)]

    results = {}
    for name, (w, du) in scenarios.items():
        ps = scenario_pd(p0, beta, du)
        e12 = ecl_loan_level(ps, F, age, bal, book["int_rate"], lgd, 12)
        elife = ecl_loan_level(ps, F, age, bal, book["int_rate"], lgd, TERM)
        ecl = np.where(stage == 1, e12, elife)
        ecl = np.where(stage == 3, lgd * ead_now, ecl)
        results[name] = {"weight": w, "delta_unemployment": du, "ecl": ecl, "ecl_12m": e12,
                         "ecl_life": elife}
    weighted = sum(r["weight"] * r["ecl"] for r in results.values())
    return {"stage": stage, "ead": ead_now, "scenarios": results, "ecl": weighted}

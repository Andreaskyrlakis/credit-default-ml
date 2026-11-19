"""Unit tests on synthetic data (they do not need the Lending Club file)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config  # noqa: E402
from src.calibration import IsotonicCalibrator, PlattCalibrator  # noqa: E402
from src.data import default_flag, emp_years, time_split  # noqa: E402
from src.ifrs9 import (balance_schedule, default_month, ecl_loan_level, empirical_lgd,  # noqa: E402
                       scenario_pd, timing_curve, window_pd)
from src.metrics import (all_metrics, bootstrap_metrics, calibration_bins,  # noqa: E402
                         expected_calibration_error, ks_statistic)
from src.models import make_model  # noqa: E402
from src.stability import psi, psi_categorical  # noqa: E402
from src.threshold import bayes_cutoff, cost_curve, expected_cost  # noqa: E402
from src.woe import WoEScorecard, fit_binning, woe_table  # noqa: E402

RNG = np.random.default_rng(0)


def synthetic_loans(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "loan_amnt": rng.uniform(1000, 35000, n), "annual_inc": rng.lognormal(11, 0.5, n),
        "dti": rng.uniform(0, 40, n), "emp_years": rng.integers(0, 11, n).astype(float),
        "fico": rng.uniform(660, 850, n), "credit_age_years": rng.uniform(1, 40, n),
        "open_acc": rng.integers(1, 30, n), "pub_rec": rng.integers(0, 3, n),
        "revol_bal": rng.lognormal(9, 1, n), "revol_util": rng.uniform(0, 100, n),
        "total_acc": rng.integers(2, 60, n), "inq_last_6mths": rng.integers(0, 6, n),
        "delinq_2yrs": rng.integers(0, 3, n), "mort_acc": rng.integers(0, 5, n),
        "pub_rec_bankruptcies": rng.integers(0, 2, n), "months_since_delinq": rng.uniform(0, 120, n),
        "home_ownership": rng.choice(["RENT", "MORTGAGE", "OWN"], n),
        "purpose": rng.choice(["debt_consolidation", "credit_card", "other"], n),
        "verification_status": rng.choice(["Verified", "Not Verified"], n),
        "region": rng.choice(["West", "South", "Midwest", "Northeast"], n),
    })
    df["loan_to_income"] = df["loan_amnt"] / df["annual_inc"]
    df.loc[rng.random(n) < 0.05, "emp_years"] = np.nan
    z = -2 - 0.02 * (df["fico"] - 750) + 0.04 * (df["dti"] - 20) + 0.3 * (df["home_ownership"] == "RENT")
    df["default"] = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(int)
    return df


# --- data -----------------------------------------------------------------------------------
def test_default_flag_and_emp_years():
    s = pd.Series(["Fully Paid", "Charged Off", "Current", "Default",
                   "Does not meet the credit policy. Status:Charged Off"])
    f = default_flag(s)
    assert f.tolist()[:2] == [0.0, 1.0] and np.isnan(f[2]) and f[3] == 1.0 and f[4] == 1.0
    e = emp_years(pd.Series(["10+ years", "< 1 year", "3 years", None]))
    assert e.tolist()[:3] == [10, 0, 3] and pd.isna(e[3])


def test_time_split_is_ordered_and_disjoint():
    dates = pd.date_range("2013-01-01", "2015-12-01", freq="MS")
    df = pd.DataFrame({"issue_d": dates, "default": 0})
    tr, va, te = time_split(df)
    assert tr["issue_d"].max() < va["issue_d"].min() <= va["issue_d"].max() < te["issue_d"].min()
    assert len(tr) + len(va) + len(te) == len(df)


# --- WoE scorecard --------------------------------------------------------------------------
def test_woe_signs_and_iv():
    bins = pd.Series(["a"] * 100 + ["b"] * 100)
    y = pd.Series([0] * 90 + [1] * 10 + [0] * 50 + [1] * 50)
    t = woe_table(bins, y)
    assert t.loc["a", "woe"] > 0 > t.loc["b", "woe"]   # safe bin positive, risky bin negative
    assert (t["iv"] >= 0).all()


def test_numeric_binning_is_monotone():
    df = synthetic_loans(20000)
    for col in ["fico", "dti"]:          # dti has 1- and 2-digit edges: tests the bin order
        fb = fit_binning(df[col], df["default"], col, max_bins=10)
        table = fb.table.drop(index="Missing", errors="ignore")
        lefts = [float(s.split(",")[0].strip("([")) for s in table.index]
        assert lefts == sorted(lefts)
        d = np.diff(table["default_rate"].to_numpy())
        assert (d <= 1e-12).all() or (d >= -1e-12).all()


def test_scorecard_points_add_up_and_pdo():
    df = synthetic_loans(5000)
    sc = WoEScorecard(config.FEATURES, min_iv=0.0).fit(df[config.FEATURES], df["default"])
    pts = sc.points_table(pdo=20, base_score=600, base_odds=50)
    row = df.iloc[[0]]
    total = sum(pts[(pts.feature == f) & (pts.bin == sc.binnings[f].labels(row[f]).iloc[0])]["points"].iloc[0]
                for f in sc.selected)
    assert total == pytest.approx(sc.score(row)[0], abs=1e-6)
    # +20 points <=> good:bad odds doubled
    p = sc.predict_proba(row)[0, 1]
    s = sc.score(row)[0]
    odds = (1 - p) / p
    assert s == pytest.approx(600 + 20 / np.log(2) * np.log(odds / 50), abs=1e-6)


# --- metrics --------------------------------------------------------------------------------
def test_metrics_perfect_and_random():
    y = np.array([0] * 50 + [1] * 50)
    perfect = all_metrics(y, np.r_[np.zeros(50) + 0.1, np.zeros(50) + 0.9])
    assert perfect["auc"] == 1 and perfect["gini"] == 1 and perfect["ks"] == 1
    rnd = all_metrics(RNG.integers(0, 2, 20000), RNG.random(20000))
    assert abs(rnd["auc"] - 0.5) < 0.02 and rnd["gini"] == pytest.approx(2 * rnd["auc"] - 1)


def test_bootstrap_interval_contains_point():
    y = RNG.integers(0, 2, 3000)
    p = np.clip(0.3 * y + 0.7 * RNG.random(3000), 0, 1)
    b = bootstrap_metrics(y, p, n_boot=100)
    for v in b.values():
        assert v["low"] <= v["value"] <= v["high"]


def test_ks_matches_definition():
    y = RNG.integers(0, 2, 2000)
    p = RNG.random(2000) + 0.2 * y
    grid = np.sort(np.unique(p))
    cdf_bad = np.searchsorted(np.sort(p[y == 1]), grid, side="right") / (y == 1).sum()
    cdf_good = np.searchsorted(np.sort(p[y == 0]), grid, side="right") / (y == 0).sum()
    assert ks_statistic(y, p) == pytest.approx(np.max(cdf_good - cdf_bad), abs=1e-9)


# --- calibration ----------------------------------------------------------------------------
def test_calibrators_fix_a_distorted_pd():
    true = RNG.uniform(0.02, 0.4, 50000)
    y = (RNG.random(50000) < true).astype(int)
    distorted = np.clip(true * 1.8, 0, 0.99)
    for cal in (PlattCalibrator(), IsotonicCalibrator()):
        fixed = cal.fit(distorted, y).transform(distorted)
        assert expected_calibration_error(y, fixed) < expected_calibration_error(y, distorted) / 3
    iso = IsotonicCalibrator().fit(distorted, y).transform(np.sort(distorted))
    assert (np.diff(iso) >= -1e-12).all()
    pred, obs = calibration_bins(y, true)
    assert np.max(np.abs(pred - obs)) < 0.03


# --- stability ------------------------------------------------------------------------------
def test_psi():
    a = RNG.normal(0, 1, 50000)
    assert psi(a, RNG.normal(0, 1, 50000)) < 0.01
    assert psi(a, RNG.normal(1, 1, 50000)) > 0.25
    assert psi_categorical(["a"] * 50 + ["b"] * 50, ["a"] * 50 + ["b"] * 50) == pytest.approx(0)


# --- IFRS 9 ---------------------------------------------------------------------------------
def test_timing_curve_and_window_pd():
    months = RNG.integers(1, 37, 5000)
    F = timing_curve(months)
    assert F[0] == 0 and F[-1] == pytest.approx(1) and (np.diff(F) >= 0).all()
    p = np.array([0.15])
    assert window_pd(p, F, np.array([0]), 36)[0] == pytest.approx(0.15)
    # 12-month PD from age 0 plus conditional rest = lifetime (survival chain rule)
    p12 = window_pd(p, F, np.array([0]), 12)[0]
    rest = window_pd(p, F, np.array([12]), 24)[0]
    assert 1 - (1 - p12) * (1 - rest) == pytest.approx(0.15)


def test_default_month():
    issue = pd.to_datetime(pd.Series(["2015-01-01", "2015-01-01", "2015-01-01"]))
    last = pd.to_datetime(pd.Series(["2015-06-01", None, "2019-01-01"]))
    assert default_month(issue, last).tolist() == [6, 1, 36]


def test_balance_schedule_is_an_annuity():
    b = balance_schedule([10000], [0.12])[0]
    assert b[0] == pytest.approx(10000) and b[-1] == pytest.approx(0, abs=1e-6)
    r = 0.01
    pay = 10000 * r / (1 - (1 + r) ** -36)
    assert b[1] == pytest.approx(10000 * (1 + r) - pay)


def test_ecl_properties():
    F = timing_curve(RNG.integers(1, 37, 5000))
    bal = balance_schedule([10000, 10000], [0.1, 0.1])
    age = np.array([0, 0])
    e0 = ecl_loan_level(np.array([0.0, 0.0]), F, age, bal, [0.1, 0.1], 0.9, 36)
    assert np.allclose(e0, 0)
    e12 = ecl_loan_level(np.array([0.2, 0.2]), F, age, bal, [0.1, 0.1], 0.9, 12)
    elife = ecl_loan_level(np.array([0.2, 0.2]), F, age, bal, [0.1, 0.1], 0.9, 36)
    assert (elife > e12).all() and (elife < 0.2 * 0.9 * 10000).all()


def test_lgd_and_scenarios():
    df = pd.DataFrame({"funded_amnt": [10000.0], "total_rec_prncp": [4000.0],
                       "recoveries": [700.0], "collection_recovery_fee": [100.0]})
    assert empirical_lgd(df)[0] == pytest.approx(0.9)
    p = np.array([0.05, 0.2])
    assert np.allclose(scenario_pd(p, 0.3, 0.0), p)
    assert (scenario_pd(p, 0.3, 2.0) > p).all() and (scenario_pd(p, 0.3, -1.0) < p).all()


# --- threshold ------------------------------------------------------------------------------
def test_bayes_cutoff_minimises_cost():
    pd_true = RNG.uniform(0, 0.5, 100000)
    y = (RNG.random(100000) < pd_true).astype(int)
    grid, cost = cost_curve(y, pd_true, c_fp=1.0, c_fn=4.0)
    assert abs(grid[np.argmin(cost)] - bayes_cutoff(1.0, 4.0)) < 0.03
    assert expected_cost(y, pd_true, 1.01, 1, 4) == pytest.approx(4 * y.mean())


# --- models ---------------------------------------------------------------------------------
@pytest.mark.parametrize("name", ["logistic", "tree", "hist_gb", "lightgbm", "mlp"])
def test_models_fit_and_rank(name):
    df = synthetic_loans(4000)
    m = make_model(name).fit(df[config.FEATURES], df["default"])
    p = m.predict_proba(df[config.FEATURES])[:, 1]
    assert ((p >= 0) & (p <= 1)).all()
    assert all_metrics(df["default"], p)["auc"] > 0.6

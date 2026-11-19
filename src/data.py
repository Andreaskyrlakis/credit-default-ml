"""Load the Lending Club file, build the default flag and the application features, split in time.

Every transformation here is row by row (no statistic is learned from the data), so it can be
applied before the split without leaking information from the test period into training.
"""

import numpy as np
import pandas as pd

from . import config

BAD_STATUSES = {"Charged Off", "Default",
                "Does not meet the credit policy. Status:Charged Off"}
GOOD_STATUSES = {"Fully Paid", "Does not meet the credit policy. Status:Fully Paid"}

REGIONS = {
    "Northeast": "CT ME MA NH RI VT NJ NY PA",
    "Midwest": "IL IN MI OH WI IA KS MN MO NE ND SD",
    "South": "DE DC FL GA MD NC SC VA WV AL KY MS TN AR LA OK TX",
    "West": "AZ CO ID MT NV NM UT WY AK CA HI OR WA",
}
STATE_TO_REGION = {s: r for r, states in REGIONS.items() for s in states.split()}


def default_flag(status: pd.Series) -> pd.Series:
    """1 = charged off / default, 0 = fully paid, NaN = loan still running at the data date."""
    flag = pd.Series(np.nan, index=status.index)
    flag[status.isin(BAD_STATUSES)] = 1.0
    flag[status.isin(GOOD_STATUSES)] = 0.0
    return flag


def emp_years(text: pd.Series) -> pd.Series:
    """'10+ years' -> 10, '< 1 year' -> 0, '3 years' -> 3, missing -> NaN."""
    t = text.astype("string").str.replace("< 1", "0", regex=False)
    return pd.to_numeric(t.str.extract(r"(\d+)")[0], errors="coerce")


def months_between(start: pd.Series, end: pd.Series) -> pd.Series:
    return (end.dt.year - start.dt.year) * 12 + (end.dt.month - start.dt.month)


def clean_chunk(raw: pd.DataFrame) -> pd.DataFrame:
    """Keep 36-month individual loans and build the modelling columns."""
    df = raw[raw["term"].astype("string").str.strip() == "36 months"].copy()
    df = df[df["application_type"].fillna("Individual") == "Individual"]
    df = df[df["annual_inc"] > 0]
    out = pd.DataFrame(index=df.index)
    out["id"] = df["id"].astype("string")
    out["issue_d"] = pd.to_datetime(df["issue_d"], format="%b-%Y")
    out["loan_status"] = df["loan_status"].astype("string")
    out[config.TARGET] = default_flag(df["loan_status"])
    out["grade"] = df["grade"].astype("string")
    out["sub_grade"] = df["sub_grade"].astype("string")
    out["int_rate"] = df["int_rate"].astype(float) / 100.0
    out["installment"] = df["installment"].astype(float)
    out["funded_amnt"] = df["funded_amnt"].astype(float)
    # application features
    out["loan_amnt"] = df["loan_amnt"].astype(float)
    out["annual_inc"] = df["annual_inc"].astype(float)
    out["loan_to_income"] = out["loan_amnt"] / out["annual_inc"]
    out["dti"] = df["dti"].astype(float)
    out["emp_years"] = emp_years(df["emp_length"])
    out["fico"] = (df["fico_range_low"] + df["fico_range_high"]) / 2.0
    earliest = pd.to_datetime(df["earliest_cr_line"], format="%b-%Y", errors="coerce")
    out["credit_age_years"] = months_between(earliest, out["issue_d"]) / 12.0
    for col in ["open_acc", "pub_rec", "revol_bal", "revol_util", "total_acc", "inq_last_6mths",
                "delinq_2yrs", "mort_acc", "pub_rec_bankruptcies"]:
        out[col] = df[col].astype(float)
    # missing = never delinquent; capped at 10 years so that "never" sits next to "long ago"
    out["months_since_delinq"] = df["mths_since_last_delinq"].astype(float).clip(upper=120).fillna(120)
    home = df["home_ownership"].astype("string")
    out["home_ownership"] = home.where(home.isin(["RENT", "MORTGAGE", "OWN"]), "OTHER")
    out["purpose"] = df["purpose"].astype("string")
    out["verification_status"] = df["verification_status"].astype("string")
    out["region"] = df["addr_state"].map(STATE_TO_REGION).fillna("Other").astype("string")
    out["addr_state"] = df["addr_state"].astype("string")
    # outcomes (never used as features): for IFRS 9 timing, LGD and the profit of the cut-off
    out["last_pymnt_d"] = pd.to_datetime(df["last_pymnt_d"], format="%b-%Y", errors="coerce")
    for col in ["total_pymnt", "total_rec_prncp", "recoveries", "collection_recovery_fee"]:
        out[col] = df[col].astype(float)
    return out


def build_loan_table(path=config.LC_FILE, chunksize=200_000) -> pd.DataFrame:
    parts = []
    reader = pd.read_csv(path, usecols=config.RAW_COLUMNS, chunksize=chunksize, low_memory=False)
    for raw in reader:
        raw = raw.dropna(subset=["issue_d", "term", "loan_status"])
        parts.append(clean_chunk(raw))
    df = pd.concat(parts, ignore_index=True)
    for col in ["loan_status", "grade", "sub_grade", "home_ownership", "purpose",
                "verification_status", "region", "addr_state"]:
        df[col] = df[col].astype("category")
    return df.sort_values(["issue_d", "id"]).reset_index(drop=True)


def load_loans() -> pd.DataFrame:
    return pd.read_parquet(config.LOANS_FILE)


def modelling_sample(df: pd.DataFrame) -> pd.DataFrame:
    """Resolved loans (paid or defaulted) issued up to the end of the test period."""
    m = df[(df["issue_d"] <= config.TEST_END) & df[config.TARGET].notna()].copy()
    m[config.TARGET] = m[config.TARGET].astype(int)
    return m


def time_split(df: pd.DataFrame):
    """Train: issued up to 2014; validation: January-June 2015; test: July-December 2015."""
    train = df[df["issue_d"] <= config.TRAIN_END]
    valid = df[(df["issue_d"] > config.TRAIN_END) & (df["issue_d"] <= config.VALID_END)]
    test = df[(df["issue_d"] > config.VALID_END) & (df["issue_d"] <= config.TEST_END)]
    return train, valid, test

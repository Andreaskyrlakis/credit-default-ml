"""Paths, dates and column lists shared by every script."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
FIGURES = ROOT / "figures"
RESULTS = ROOT / "results"

LC_URL = ("https://huggingface.co/datasets/codesignal/lending-club-loan-accepted/"
          "resolve/main/accepted_2007_to_2018Q4.csv")
LC_FILE = RAW / "accepted_2007_to_2018Q4.csv"
LOANS_FILE = PROCESSED / "loans_36m.parquet"
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={}"

SEED = 42

# Time split on the issue date (36-month loans, all matured by the end of the data in 2018Q4)
TRAIN_END = "2014-12-31"
VALID_END = "2015-06-30"
TEST_END = "2015-12-31"

# IFRS 9 reporting date for the test portfolio (loans issued July-December 2015)
REPORTING_DATE = "2016-06-30"

# Columns read from the raw file
RAW_COLUMNS = [
    "id", "issue_d", "term", "loan_status", "grade", "sub_grade", "int_rate", "installment",
    "funded_amnt", "loan_amnt", "annual_inc", "dti", "emp_length", "home_ownership", "purpose",
    "addr_state", "verification_status", "fico_range_low", "fico_range_high", "earliest_cr_line",
    "open_acc", "pub_rec", "revol_bal", "revol_util", "total_acc", "inq_last_6mths",
    "delinq_2yrs", "mort_acc", "pub_rec_bankruptcies", "mths_since_last_delinq",
    "application_type", "last_pymnt_d", "total_pymnt", "total_rec_prncp", "recoveries",
    "collection_recovery_fee",
]

# Features known when the loan is applied for. Lending Club's own grade and interest rate are
# deliberately left out: they are the output of Lending Club's model, used here as a benchmark.
NUMERIC_FEATURES = [
    "loan_amnt", "annual_inc", "loan_to_income", "dti", "emp_years", "fico", "credit_age_years",
    "open_acc", "pub_rec", "revol_bal", "revol_util", "total_acc", "inq_last_6mths",
    "delinq_2yrs", "mort_acc", "pub_rec_bankruptcies", "months_since_delinq",
]
CATEGORICAL_FEATURES = ["home_ownership", "purpose", "verification_status", "region"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "default"

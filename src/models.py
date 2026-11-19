"""The candidate models and their hyperparameter search spaces.

All imputation, scaling and encoding sit inside a scikit-learn Pipeline, so they are learned on
the training folds only.
"""

import numpy as np
from lightgbm import LGBMClassifier
from scipy.stats import loguniform, randint, uniform
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (HistGradientBoostingClassifier, RandomForestClassifier,
                              VotingClassifier)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from . import config

# Heavy-tailed amounts are put on a log scale before scaling (for the linear model and the MLP).
LOG_FEATURES = ["loan_amnt", "annual_inc", "loan_to_income", "revol_bal"]


def _log1p_columns(X):
    X = X.copy()
    for c in LOG_FEATURES:
        X[c] = np.log1p(X[c].clip(lower=0))
    return X


def preprocessor(scale: bool) -> ColumnTransformer:
    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("scale", StandardScaler()))
    numeric = Pipeline(num_steps)
    cat = OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=0.005,
                        sparse_output=False)
    ct = ColumnTransformer([("num", numeric, config.NUMERIC_FEATURES),
                            ("cat", cat, config.CATEGORICAL_FEATURES)])
    if scale:
        return Pipeline([("log", FunctionTransformer(_log1p_columns)), ("ct", ct)])
    return ct


def make_model(name: str, **params):
    seed = config.SEED
    if name == "logistic":
        est, scale = LogisticRegression(max_iter=2000, C=params.get("C", 1.0)), True
    elif name == "tree":
        est, scale = DecisionTreeClassifier(random_state=seed, **params), False
    elif name == "random_forest":
        p = {"n_estimators": 300, "min_samples_leaf": 50, "max_features": 0.3, **params}
        est, scale = RandomForestClassifier(random_state=seed, n_jobs=4, **p), False
    elif name == "hist_gb":
        est, scale = HistGradientBoostingClassifier(random_state=seed, **params), False
    elif name == "lightgbm":
        p = {"n_estimators": 400, "learning_rate": 0.03, "num_leaves": 31, **params}
        est, scale = LGBMClassifier(random_state=seed, n_jobs=4, verbose=-1, **p), False
    elif name == "mlp":
        p = {"hidden_layer_sizes": (64, 32), "alpha": 1e-3, **params}
        est, scale = MLPClassifier(random_state=seed, early_stopping=True, max_iter=200,
                                   learning_rate_init=1e-3, batch_size=512, **p), True
    else:
        raise ValueError(name)
    return Pipeline([("prep", preprocessor(scale)), ("model", est)])


def make_voting(estimators: dict) -> VotingClassifier:
    """Soft vote (average of predicted PDs) of already-configured pipelines."""
    return VotingClassifier(list(estimators.items()), voting="soft", n_jobs=1)


def predict_named(name: str, fitted: dict, scorecard, X):
    """PD of one named model ('vote' = mean of logistic, LightGBM and MLP)."""
    if name == "scorecard":
        return scorecard.predict_proba(X)[:, 1]
    if name == "vote":
        return np.mean([fitted[k].predict_proba(X)[:, 1] for k in ("logistic", "lightgbm", "mlp")], 0)
    return fitted[name].predict_proba(X)[:, 1]


SEARCH_SPACES = {
    "logistic": {"model__C": loguniform(1e-3, 10)},
    "tree": {"model__max_depth": randint(3, 12), "model__min_samples_leaf": randint(50, 2000)},
    "random_forest": {"model__max_depth": randint(6, 20),
                      "model__min_samples_leaf": randint(20, 300),
                      "model__max_features": uniform(0.1, 0.5)},
    "hist_gb": {"model__learning_rate": loguniform(0.02, 0.3),
                "model__max_leaf_nodes": randint(8, 64),
                "model__min_samples_leaf": randint(50, 1000),
                "model__l2_regularization": loguniform(1e-3, 10)},
    "lightgbm": {"model__learning_rate": loguniform(0.01, 0.1),
                 "model__num_leaves": randint(8, 64),
                 "model__min_child_samples": randint(50, 1000),
                 "model__subsample": uniform(0.6, 0.4), "model__subsample_freq": [1],
                 "model__colsample_bytree": uniform(0.5, 0.5),
                 "model__reg_lambda": loguniform(1e-3, 10),
                 "model__n_estimators": randint(200, 800)},
    "mlp": {"model__hidden_layer_sizes": [(32,), (64,), (64, 32), (128, 64)],
            "model__alpha": loguniform(1e-5, 1e-1)},
}

N_ITER = {"logistic": 10, "tree": 15, "random_forest": 8, "hist_gb": 15, "lightgbm": 20, "mlp": 8}

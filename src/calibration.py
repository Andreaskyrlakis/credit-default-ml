"""Recalibrating predicted PDs on the validation set: Platt scaling and isotonic regression."""

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


class PlattCalibrator:
    """Logistic regression of the outcome on logit(PD): fixes level and slope."""

    def fit(self, p, y):
        self.lr = LogisticRegression(C=1e6, max_iter=1000).fit(_logit(p).reshape(-1, 1), y)
        return self

    def transform(self, p):
        return self.lr.predict_proba(_logit(p).reshape(-1, 1))[:, 1]

    @property
    def slope_intercept(self):
        return float(self.lr.coef_[0, 0]), float(self.lr.intercept_[0])


class IsotonicCalibrator:
    """Monotone step function from PD to observed default rate."""

    def fit(self, p, y):
        self.iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-4, y_max=1 - 1e-4).fit(p, y)
        return self

    def transform(self, p):
        return self.iso.predict(p)

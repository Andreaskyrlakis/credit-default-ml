"""Choosing the approval cut-off on the PD.

Two views:
- cost view: refusing a good client costs c_fp (lost margin), accepting a defaulter costs c_fn
  (lost principal); minimise c_fp * FP + c_fn * FN. The Bayes-optimal cut-off for calibrated PDs
  is PD* = c_fp / (c_fp + c_fn).
- profit view: realised profit of each loan = total amount repaid - amount lent (from the data);
  approve every loan with PD below the cut-off and add up the realised profit.
"""

import numpy as np


def expected_cost(y, p, cutoff, c_fp, c_fn) -> float:
    """Average cost per applicant when loans with PD >= cutoff are refused."""
    y, p = np.asarray(y), np.asarray(p)
    refuse = p >= cutoff
    fp = np.sum(refuse & (y == 0))     # good client refused
    fn = np.sum(~refuse & (y == 1))    # defaulter accepted
    return float((c_fp * fp + c_fn * fn) / len(y))


def cost_curve(y, p, c_fp, c_fn, grid=None):
    grid = np.linspace(0.0, 0.6, 121) if grid is None else grid
    return grid, np.array([expected_cost(y, p, c, c_fp, c_fn) for c in grid])


def bayes_cutoff(c_fp, c_fn) -> float:
    return c_fp / (c_fp + c_fn)


def profit_curve(p, profit, grid=None):
    """Total realised profit and approval rate when approving PD < cutoff."""
    p, profit = np.asarray(p), np.asarray(profit)
    grid = np.linspace(0.0, 0.6, 121) if grid is None else grid
    total = np.array([profit[p < c].sum() for c in grid])
    approval = np.array([(p < c).mean() for c in grid])
    return grid, total, approval


def best_cutoff(grid, values, maximise=True) -> float:
    i = int(np.argmax(values) if maximise else np.argmin(values))
    return float(grid[i])

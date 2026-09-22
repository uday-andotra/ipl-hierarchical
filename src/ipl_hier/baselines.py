"""Non-hierarchical baselines for the holdout season.

These are the comparisons a referee will ask for. None use the
season-specific alpha of the holdout year.
"""
from __future__ import annotations

import numpy as np

from .metrics import brier, log_loss


def coin(y):
    p = np.full(len(y), 0.5)
    return {"name": "coin", "brier": brier(y, p), "log_loss": log_loss(y, p), "acc": float(np.mean(y == 1) * 0 + 0.5)}


def always_team1(y):
    p = np.ones(len(y))
    return {
        "name": "always_team1",
        "brier": brier(y, p),
        "log_loss": log_loss(y, p),
        "acc": float(np.mean(y == 1)),
    }


def empirical_rate(y_train, n_test):
    r = float(np.mean(y_train))
    p = np.full(n_test, r)
    return r, p


def pooled_bt_eta(t1, t2, y, home=None, n_teams=None):
    """Identified logit least squares on train, no season layer."""
    n = len(y)
    k = int(n_teams)
    # columns: mu_0 .. mu_{k-2}, optional home
    cols = k - 1 + (1 if home is not None else 0)
    X = np.zeros((n, cols))
    for i in range(n):
        a, b = int(t1[i]), int(t2[i])
        if a < k - 1:
            X[i, a] += 1.0
        else:
            X[i, : k - 1] -= 1.0
        if b < k - 1:
            X[i, b] -= 1.0
        else:
            X[i, : k - 1] += 1.0
        if home is not None:
            X[i, -1] = home[i]
    # ridge on the logit working response via IRLS-lite: one Newton on 0.5 start
    p = np.full(n, 0.5)
    w = p * (1 - p)
    z = (y - p) / np.clip(w, 1e-6, None)
    Xw = X * np.sqrt(w)[:, None]
    zw = z * np.sqrt(w)
    coef, *_ = np.linalg.lstsq(Xw, zw, rcond=None)
    mu = np.zeros(k)
    mu[: k - 1] = coef[: k - 1]
    mu[k - 1] = -mu[: k - 1].sum()
    bh = float(coef[-1]) if home is not None else 0.0
    return mu, bh


def sigmoid(eta):
    return 1.0 / (1.0 + np.exp(-np.clip(eta, -30, 30)))


def score(name, y, p):
    return {
        "name": name,
        "brier": brier(y, p),
        "log_loss": log_loss(y, p),
        "acc": float(np.mean((p > 0.5) == y)),
        "mean_p": float(np.mean(p)),
    }

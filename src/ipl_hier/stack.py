"""Second-stage layers on an already-fit p.

Fit on past matches only. Freeze on the holdout season.
Not a new strength model. Not a GNN.
"""
from __future__ import annotations

import numpy as np

from .metrics import brier, log_loss


def logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def expit(eta):
    return 1.0 / (1.0 + np.exp(-np.clip(eta, -30, 30)))


def blend(p_mu, p_recent, lam):
    """lam=1 is franchise mu only; lam=0 is last-season alpha."""
    return lam * np.asarray(p_mu, float) + (1.0 - lam) * np.asarray(p_recent, float)


def choose_lambda(y, p_mu, p_recent, grid=None):
    if grid is None:
        grid = np.linspace(0.0, 1.0, 11)
    y = np.asarray(y, float)
    best, best_l = 1e9, 1.0
    for lam in grid:
        s = brier(y, blend(p_mu, p_recent, lam))
        if s < best:
            best, best_l = s, float(lam)
    return best_l, best


def fit_platt(y, p):
    """p' = expit(a + b logit(p)). One Newton / IRLS step is enough here."""
    z = logit(p)
    y = np.asarray(y, float)
    X = np.column_stack([np.ones(len(y)), z])
    beta = np.zeros(2)
    for _ in range(12):
        eta = X @ beta
        pr = expit(eta)
        w = pr * (1 - pr)
        grad = X.T @ (y - pr)
        H = X.T @ (X * w[:, None])
        H = H + 1e-4 * np.eye(2)
        beta = beta + np.linalg.solve(H, grad)
    return float(beta[0]), float(beta[1])


def apply_platt(p, a, b):
    return expit(a + b * logit(p))


def residual_logit(y, p, extras):
    """Tiny extra layer: logit(p) plus a few pre-match columns (home, chase).

    extras: (n, k) array. Ridge on the working logit. Not a neural net.
    """
    y = np.asarray(y, float)
    z0 = logit(p)
    X = np.column_stack([np.ones(len(y)), z0, extras])
    # one-step logistic ridge
    pr = expit(z0)
    w = np.clip(pr * (1 - pr), 1e-4, None)
    target = z0 + (y - pr) / w
    xtx = X.T @ (X * w[:, None]) + 0.5 * np.eye(X.shape[1])
    xty = X.T @ (target * w)
    coef = np.linalg.solve(xtx, xty)
    return coef


def apply_residual(p, extras, coef):
    z0 = logit(p)
    X = np.column_stack([np.ones(len(p)), z0, extras])
    return expit(X @ coef)


def report(y, p, name):
    return {
        "name": name,
        "brier": brier(y, p),
        "log_loss": log_loss(y, p),
        "acc": float(np.mean((np.asarray(p) > 0.5) == np.asarray(y))),
        "mean_p": float(np.mean(p)),
    }


def superlearner(y, P):
    """Nonnegative weights on columns of P that sum to 1. Discrete search.

    P is (n, m): candidate probabilities already locked at train time.
    """
    y = np.asarray(y, float)
    P = np.asarray(P, float)
    m = P.shape[1]
    grid = np.linspace(0.0, 1.0, 6)
    best_w, best = None, 1e9

    def rec(i, left, acc):
        nonlocal best_w, best
        if i == m - 1:
            w = np.array(acc + [left], dtype=float)
            s = brier(y, np.clip(P @ w, 0, 1))
            if s < best:
                best, best_w = s, w
            return
        for g in grid:
            if g <= left + 1e-12:
                rec(i + 1, left - g, acc + [g])

    rec(0, 1.0, [])
    return best_w, float(best)

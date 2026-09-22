"""Implied win probabilities from a posted price. Rival only — not a column in eta."""

from __future__ import annotations

import numpy as np


def decimal_to_p(odds):
    o = np.asarray(odds, dtype=float)
    p = np.full_like(o, np.nan, dtype=float)
    ok = o > 1.0
    p[ok] = 1.0 / o[ok]
    return p


def two_way_normalize(p1, p2):
    """Strip overround so p1+p2=1 when both sides present."""
    a = np.asarray(p1, dtype=float)
    b = np.asarray(p2, dtype=float)
    s = a + b
    out = np.full_like(a, np.nan)
    ok = s > 0
    out[ok] = a[ok] / s[ok]
    return out


def score_vs_market(y, p_model, p_mkt):
    from .metrics import brier, log_loss

    y = np.asarray(y, dtype=float)
    p_model = np.asarray(p_model, dtype=float)
    p_mkt = np.asarray(p_mkt, dtype=float)
    ok = np.isfinite(p_mkt) & np.isfinite(p_model)
    if not ok.any():
        return {"n_market": 0}
    return {
        "n_market": int(ok.sum()),
        "model_brier": brier(y[ok], p_model[ok]),
        "market_brier": brier(y[ok], p_mkt[ok]),
        "brier_minus_market": float(brier(y[ok], p_model[ok]) - brier(y[ok], p_mkt[ok])),
        "model_log_loss": log_loss(y[ok], p_model[ok]),
        "market_log_loss": log_loss(y[ok], p_mkt[ok]),
    }

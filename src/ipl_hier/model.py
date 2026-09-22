"""Identified hierarchical logit Bradley-Terry.

P(team1 beats team2 in season s) =
    logit^{-1}( α_{1,s} - α_{2,s} + β_home * home_edge + β_chase * 1{team1 chasing} )

α_{i,s} ~ N(μ_i, σ_α²)
μ_i ~ N(0, 1), sum-to-zero across franchises.

No spread, no EPA pile. Same linear predictor at train and predict time.
"""

from __future__ import annotations

import numpy as np
import pymc as pm
import pytensor.tensor as pt


def build_model(
    dm: dict,
    include_chase: bool = True,
    include_home: bool = True,
    include_venue: bool = False,
) -> pm.Model:
    t1, t2, season = dm["t1"], dm["t2"], dm["season"]
    y, home, chase = dm["y"], dm["home"], dm["chase"]
    n_teams, n_seasons = dm["n_teams"], dm["n_seasons"]

    with pm.Model() as model:
        mu_raw = pm.Normal("mu_raw", 0.0, 1.0, shape=n_teams)
        mu = pm.Deterministic("mu", mu_raw - pt.mean(mu_raw))

        sigma_alpha = pm.HalfNormal("sigma_alpha", 0.5)
        alpha_raw = pm.Normal("alpha_raw", 0.0, 1.0, shape=(n_teams, n_seasons))
        alpha = pm.Deterministic("alpha", mu[:, None] + sigma_alpha * alpha_raw)

        beta_home = pm.Normal("beta_home", 0.0, 0.5) if include_home else 0.0
        beta_chase = pm.Normal("beta_chase", 0.0, 0.5) if include_chase else 0.0

        eta = alpha[t1, season] - alpha[t2, season]
        if include_home:
            eta = eta + beta_home * home
        if include_chase:
            eta = eta + beta_chase * chase
        if include_venue and "city" in dm:
            sigma_v = pm.HalfNormal("sigma_venue", 0.3)
            v_raw = pm.Normal("venue_raw", 0.0, 1.0, shape=dm["n_cities"])
            venue = pm.Deterministic("venue", sigma_v * (v_raw - pt.mean(v_raw)))
            eta = eta + venue[dm["city"]]

        pm.Bernoulli("y", p=pm.math.invlogit(eta), observed=y)
    return model


def posterior_predictive_proba(idata, dm, include_chase=True, include_home=True):
    """P(team1 wins) from the same draws used for strength ranks."""
    alpha = idata.posterior["alpha"].values  # chain, draw, team, season
    a = alpha.reshape(-1, alpha.shape[-2], alpha.shape[-1])
    t1, t2, s = dm["t1"], dm["t2"], dm["season"]
    eta = a[:, t1, s] - a[:, t2, s]
    if include_home and "beta_home" in idata.posterior:
        bh = idata.posterior["beta_home"].values.reshape(-1)[:, None]
        eta = eta + bh * dm["home"]
    if include_chase and "beta_chase" in idata.posterior:
        bc = idata.posterior["beta_chase"].values.reshape(-1)[:, None]
        eta = eta + bc * dm["chase"]
    if "venue" in idata.posterior and "city" in dm:
        v = idata.posterior["venue"].values
        v = v.reshape(-1, v.shape[-1])
        eta = eta + v[:, dm["city"]]
    p = 1.0 / (1.0 + np.exp(-eta))
    return p


def summary_match_table(tape, p_draws: np.ndarray) -> "object":
    import pandas as pd

    p_mean = p_draws.mean(axis=0)
    p_lo = np.quantile(p_draws, 0.05, axis=0)
    p_hi = np.quantile(p_draws, 0.95, axis=0)
    df = tape.frame.copy()
    df["p_team1"] = p_mean
    df["p_05"] = p_lo
    df["p_95"] = p_hi
    df["pred_team1"] = (p_mean > 0.5).astype(int)
    return df


from .metrics import brier, log_loss  # noqa: F401  (re-export)

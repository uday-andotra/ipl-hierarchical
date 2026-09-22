from __future__ import annotations

import numpy as np

from .data import MatchTape, design_matrices


def proba_from_mu(idata, tape: MatchTape) -> np.ndarray:
    """Out-of-sample season: use franchise mean μ, not a fitted α_{s*}."""
    mu = idata.posterior["mu"].values.reshape(-1, idata.posterior["mu"].values.shape[-1])
    dm = design_matrices(tape)
    t1, t2 = dm["t1"], dm["t2"]
    eta = mu[:, t1] - mu[:, t2]
    if "beta_home" in idata.posterior:
        eta = eta + idata.posterior["beta_home"].values.reshape(-1)[:, None] * dm["home"]
    if "beta_chase" in idata.posterior:
        eta = eta + idata.posterior["beta_chase"].values.reshape(-1)[:, None] * dm["chase"]
    if "venue" in idata.posterior and "city" in dm:
        v = idata.posterior["venue"].values
        v = v.reshape(-1, v.shape[-1])
        city = np.asarray(dm["city"], dtype=int)
        city = np.clip(city, 0, v.shape[-1] - 1)
        eta = eta + v[:, city]
    return 1.0 / (1.0 + np.exp(-eta))


def team_ranks(idata, teams: list[str]) -> "object":
    import pandas as pd

    mu = idata.posterior["mu"].values.reshape(-1, len(teams))
    return (
        pd.DataFrame(
            {
                "team": teams,
                "mu_mean": mu.mean(0),
                "mu_05": np.quantile(mu, 0.05, axis=0),
                "mu_95": np.quantile(mu, 0.95, axis=0),
            }
        )
        .sort_values("mu_mean", ascending=False)
        .reset_index(drop=True)
    )

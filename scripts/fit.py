#!/usr/bin/env python3
"""Fit the hierarchical BT model and write a holdout report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pymc as pm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ipl_hier.data import clean_matches, encode, design_matrices, holdout_last_season, load_raw
from ipl_hier.metrics import brier, log_loss
from ipl_hier.market import decimal_to_p, score_vs_market, two_way_normalize
from ipl_hier.model import build_model, posterior_predictive_proba
from ipl_hier.predict import proba_from_mu, team_ranks


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=str(ROOT / "data" / "matches_raw.csv"))
    p.add_argument("--draws", type=int, default=800)
    p.add_argument("--tune", type=int, default=800)
    p.add_argument("--chains", type=int, default=2)
    p.add_argument("--out", default=str(ROOT / "reports"))
    p.add_argument("--venue", action="store_true")
    p.add_argument("--odds-team1", default="")
    p.add_argument("--odds-team2", default="")
    p.add_argument("--holdout-season", default="", help="e.g. 2025; default is last season in the file")
    args = p.parse_args()

    raw = load_raw(args.data)
    full = encode(clean_matches(raw))
    train_df, test_df, last = holdout_last_season(
        full, season=args.holdout_season or None
    )
    train = encode(train_df)
    # keep team/season indices aligned with the training tape only
    test_df = test_df.loc[
        test_df["team1"].isin(train.teams) & test_df["team2"].isin(train.teams)
    ].copy()
    test = encode(test_df)
    # remap test teams onto train indices
    test.team_index = train.team_index
    test.teams = train.teams

    dm_tr = design_matrices(train)
    model = build_model(dm_tr, include_venue=args.venue)
    with model:
        idata = pm.sample(
            draws=args.draws,
            tune=args.tune,
            chains=args.chains,
            target_accept=0.9,
            random_seed=42,
            progressbar=True,
        )

    p_in = posterior_predictive_proba(idata, dm_tr)
    p_in_mean = p_in.mean(axis=0)
    p_oos = proba_from_mu(idata, test)
    p_oos_mean = p_oos.mean(axis=0)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    ranks = team_ranks(idata, train.teams)
    ranks.to_csv(out / "franchise_strength.csv", index=False)

    test_out = test.frame.copy()
    test_out["p_team1"] = p_oos_mean
    test_out["p_05"] = np.quantile(p_oos, 0.05, axis=0)
    test_out["p_95"] = np.quantile(p_oos, 0.95, axis=0)
    test_out.to_csv(out / f"holdout_{last}.csv", index=False)

    metrics = {
        "holdout_season": last,
        "n_train": int(len(train.frame)),
        "n_test": int(len(test.frame)),
        "in_sample_accuracy": float(np.mean((p_in_mean > 0.5) == dm_tr["y"])),
        "in_sample_brier": brier(dm_tr["y"], p_in_mean),
        "in_sample_log_loss": log_loss(dm_tr["y"], p_in_mean),
        "holdout_accuracy": float(np.mean((p_oos_mean > 0.5) == test.frame["y"].to_numpy())),
        "holdout_brier": brier(test.frame["y"].to_numpy(), p_oos_mean),
        "holdout_log_loss": log_loss(test.frame["y"].to_numpy(), p_oos_mean),
        "holdout_base_rate": float(test.frame["y"].mean()),
        "mean_holdout_p": float(p_oos_mean.mean()),
        "beta_home_mean": float(idata.posterior["beta_home"].values.mean()),
        "beta_chase_mean": float(idata.posterior["beta_chase"].values.mean()),
        "sigma_alpha_mean": float(idata.posterior["sigma_alpha"].values.mean()),
        "venue": bool(args.venue),
        "note": "odds columns are a rival forecast; they are not inside eta",
    }
    if args.odds_team1 and args.odds_team1 in test.frame.columns:
        p1 = decimal_to_p(test.frame[args.odds_team1])
        if args.odds_team2 and args.odds_team2 in test.frame.columns:
            p2 = decimal_to_p(test.frame[args.odds_team2])
            p_mkt = two_way_normalize(p1, p2)
        else:
            p_mkt = p1
        test_out["p_market"] = p_mkt
        test_out.to_csv(out / f"holdout_{last}.csv", index=False)
        metrics.update(score_vs_market(test.frame["y"].to_numpy(), p_oos_mean, p_mkt))
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    print(ranks.head(10).to_string(index=False))


if __name__ == "__main__":
    main()

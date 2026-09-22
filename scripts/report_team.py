#!/usr/bin/env python3
"""League fit, one-franchise report. Not a one-team model."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ipl_hier.metrics import brier, log_loss
from ipl_hier.names import canon_team


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--team", required=True)
    p.add_argument("--reports", default=str(ROOT / "reports"))
    args = p.parse_args()

    team = canon_team(args.team) or args.team.strip()
    rep = Path(args.reports)
    ranks = pd.read_csv(rep / "franchise_strength.csv")
    hold_files = sorted(rep.glob("holdout_*.csv"))
    if not hold_files:
        raise SystemExit("run scripts/fit.py first")
    hold = pd.read_csv(hold_files[-1])

    row = ranks.loc[ranks["team"].str.lower() == team.lower()]
    if row.empty:
        row = ranks.loc[ranks["team"].str.lower().str.contains(team.lower())]
    if row.empty:
        raise SystemExit("unknown team")
    row = row.iloc[0]
    team = str(row["team"])

    sl = hold.loc[(hold["team1"] == team) | (hold["team2"] == team)].copy()
    if sl.empty:
        raise SystemExit(f"no holdout games for {team}")

    p_win = sl["p_team1"].to_numpy(dtype=float).copy()
    y_win = sl["y"].to_numpy(dtype=float).copy()
    flip = sl["team2"].eq(team).to_numpy()
    p_win[flip] = 1.0 - sl.loc[flip, "p_team1"].to_numpy()
    y_win[flip] = 1.0 - sl.loc[flip, "y"].to_numpy()
    sl["p_this_team"] = p_win
    sl["this_team_won"] = y_win
    sl["opponent"] = sl["team2"].where(sl["team1"].eq(team), sl["team1"])

    ranks2 = ranks.reset_index(drop=True)
    rank = int(ranks2.index[ranks2["team"] == team][0] + 1)
    out = {
        "team": team,
        "rank": rank,
        "mu_mean": float(row["mu_mean"]),
        "mu_05": float(row["mu_05"]),
        "mu_95": float(row["mu_95"]),
        "n_holdout_games": int(len(sl)),
        "holdout_wins": int(y_win.sum()),
        "mean_p": float(p_win.mean()),
        "brier": brier(y_win, p_win),
        "log_loss": log_loss(y_win, p_win),
        "acc": float(((p_win > 0.5) == y_win).mean()),
        "note": "League-wide fit; this file is a filter.",
    }
    dest = rep / f"team_{team.replace(' ', '_')}.csv"
    sl.to_csv(dest, index=False)
    print(json.dumps(out, indent=2))
    print("wrote", dest)


if __name__ == "__main__":
    main()

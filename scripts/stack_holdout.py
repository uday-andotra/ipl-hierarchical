#!/usr/bin/env python3
"""Apply Platt calibration on a holdout file from fit.py.

Usage:
  python scripts/stack_holdout.py --holdout reports/holdout_2026.csv

Optional extras: home_edge, team1_chasing columns if present.
Split: first 70% of the file to fit the layer, last 30% to score.
This is only honest if you later switch to a previous season for the fit.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ipl_hier.stack import apply_platt, apply_residual, fit_platt, report, residual_logit, superlearner


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--holdout", required=True)
    p.add_argument("--frac", type=float, default=0.7)
    args = p.parse_args()

    df = pd.read_csv(args.holdout)
    y = df["y"].to_numpy(dtype=float)
    pr = df["p_team1"].to_numpy(dtype=float)
    n = len(df)
    cut = max(int(n * args.frac), 20)
    extras_cols = [c for c in ("home_edge", "team1_chasing") if c in df.columns]
    extras = df[extras_cols].to_numpy(dtype=float) if extras_cols else None

    a, b = fit_platt(y[:cut], pr[:cut])
    p_pl = apply_platt(pr, a, b)

    out = {
        "n": n,
        "fit_rows": cut,
        "platt_a": a,
        "platt_b": b,
        "base": report(y[cut:], pr[cut:], "raw"),
        "platt": report(y[cut:], p_pl[cut:], "platt"),
    }
    cands = [pr, p_pl, np.full(n, 0.5)]
    names = ["raw", "platt", "coin"]
    if extras is not None:
        coef = residual_logit(y[:cut], pr[:cut], extras[:cut])
        p_res = apply_residual(pr, extras, coef)
        out["residual_coef"] = coef.tolist()
        out["residual"] = report(y[cut:], p_res[cut:], "residual_logit")
        cands.append(p_res)
        names.append("residual")
    P = np.column_stack(cands)
    w, sl_brier_fit = superlearner(y[:cut], P[:cut])
    p_sl = np.clip(P @ w, 0, 1)
    out["superlearner_weights"] = dict(zip(names, [float(x) for x in w]))
    out["superlearner_fit_brier"] = sl_brier_fit
    out["superlearner"] = report(y[cut:], p_sl[cut:], "superlearner")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

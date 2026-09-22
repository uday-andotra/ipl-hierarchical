#!/usr/bin/env python3
"""Parse Cricsheet IPL csv2 zip into data/matches_raw.csv."""

from __future__ import annotations

import argparse
import csv
import io
import zipfile
from collections import defaultdict
from pathlib import Path


def parse_info(text: str) -> dict:
    rows = list(csv.reader(io.StringIO(text)))
    d = defaultdict(list)
    for r in rows:
        if not r or r[0] != "info":
            continue
        key = r[1]
        d[key].append(r[2] if len(r) > 2 else "")
    def one(k, default=""):
        v = d.get(k, [default])
        return v[0] if v else default

    teams = d.get("team", [])
    return {
        "match_id": one("match_id"),
        "season": one("season"),
        "date": one("date"),
        "venue": one("venue"),
        "city": one("city"),
        "team1": teams[0] if len(teams) > 0 else "",
        "team2": teams[1] if len(teams) > 1 else "",
        "toss_winner": one("toss_winner"),
        "toss_decision": one("toss_decision"),
        "winner": one("winner"),
        "winner_runs": one("winner_runs"),
        "winner_wickets": one("winner_wickets"),
        "outcome": one("outcome"),
        "method": one("method"),
        "event": one("event"),
        "match_number": one("match_number"),
        "gender": one("gender"),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--zip", required=True)
    p.add_argument("--out", default="data/matches_raw.csv")
    args = p.parse_args()
    z = zipfile.ZipFile(args.zip)
    recs = [
        parse_info(z.read(n).decode("utf-8", "replace"))
        for n in z.namelist()
        if n.endswith("_info.csv")
    ]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(recs[0].keys()))
        w.writeheader()
        w.writerows(recs)
    print(f"wrote {len(recs)} matches -> {out}")


if __name__ == "__main__":
    main()

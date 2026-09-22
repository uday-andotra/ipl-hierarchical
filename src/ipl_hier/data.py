from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .names import CITY_HOME, canon_team


@dataclass
class MatchTape:
    frame: pd.DataFrame
    teams: list[str]
    seasons: list[str]
    team_index: dict[str, int]
    season_index: dict[str, int]


def load_raw(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def clean_matches(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    for col in ("team1", "team2", "toss_winner", "winner"):
        df[col] = df[col].fillna("").map(canon_team)

    df["season"] = df["season"].astype(str)
    df["city"] = df["city"].fillna("").str.strip()
    df["toss_decision"] = df["toss_decision"].fillna("").str.lower()

    decided = df["winner"].ne("") & df["winner"].isin(df["team1"]) | df["winner"].isin(df["team2"])
    decided = df["winner"].ne("") & (
        (df["winner"] == df["team1"]) | (df["winner"] == df["team2"])
    )
    df = df.loc[decided].copy()

    df["y"] = (df["winner"] == df["team1"]).astype(int)

    home = df["city"].str.lower().map(CITY_HOME)
    df["team1_home"] = (home == df["team1"]).astype(int)
    df["team2_home"] = (home == df["team2"]).astype(int)
    # +1 if team1 is at a mapped home venue, -1 if team2 is, else 0
    df["home_edge"] = df["team1_home"] - df["team2_home"]

    chase_team = np.where(
        df["toss_decision"].eq("field"),
        df["toss_winner"],
        np.where(df["team1"].eq(df["toss_winner"]), df["team2"], df["team1"]),
    )
    df["team1_chasing"] = (chase_team == df["team1"]).astype(int)
    return df.reset_index(drop=True)


def encode(df: pd.DataFrame) -> MatchTape:
    teams = sorted(set(df["team1"]).union(df["team2"]))
    seasons = sorted(df["season"].unique(), key=_season_key)
    return MatchTape(
        frame=df,
        teams=teams,
        seasons=seasons,
        team_index={t: i for i, t in enumerate(teams)},
        season_index={s: i for i, s in enumerate(seasons)},
    )


def _season_key(s: str):
    # "2007/08" sorts before "2009"
    head = s.split("/")[0]
    return int(head)


def design_matrices(tape: MatchTape, city_index: dict | None = None):
    df = tape.frame
    t1 = df["team1"].map(tape.team_index).to_numpy()
    t2 = df["team2"].map(tape.team_index).to_numpy()
    s = df["season"].map(tape.season_index).to_numpy()
    y = df["y"].to_numpy()
    home = df["home_edge"].to_numpy()
    chase = df["team1_chasing"].to_numpy()
    if city_index is None:
        cities = sorted(df["city"].fillna("").astype(str).unique())
        city_index = {c: i for i, c in enumerate(cities)}
    cities = list(city_index.keys())
    return {
        "t1": t1,
        "t2": t2,
        "season": s,
        "y": y,
        "home": home,
        "chase": chase,
        "city": df["city"].fillna("").astype(str).map(city_index).to_numpy(),
        "n_teams": len(tape.teams),
        "n_seasons": len(tape.seasons),
        "n_cities": len(cities),
        "cities": cities,
    }


def holdout_last_season(tape: MatchTape, season: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    last = season if season is not None else tape.seasons[-1]
    if last not in set(tape.seasons) and last not in set(tape.frame["season"].astype(str)):
        raise ValueError(f"season {last} not in tape: {tape.seasons[-6:]}")
    train = tape.frame.loc[tape.frame["season"].astype(str) != str(last)].copy()
    # do not train on future seasons after the holdout year
    def _key(s):
        return int(str(s).split("/")[0])
    train = train.loc[train["season"].map(_key) < _key(last)].copy()
    test = tape.frame.loc[tape.frame["season"].astype(str) == str(last)].copy()
    return train, test, str(last)

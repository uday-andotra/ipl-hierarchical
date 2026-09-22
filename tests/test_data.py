from pathlib import Path
import pandas as pd
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ipl_hier.data import clean_matches, encode, design_matrices
from ipl_hier.names import canon_team


def test_aliases():
    assert canon_team("Delhi Daredevils") == "Delhi Capitals"
    assert canon_team("Kings XI Punjab") == "Punjab Kings"
    assert canon_team("Royal Challengers Bangalore") == "Royal Challengers Bengaluru"


def test_clean_drops_no_result_and_encodes_y():
    raw = pd.DataFrame(
        {
            "team1": ["Mumbai Indians", "Chennai Super Kings"],
            "team2": ["Kolkata Knight Riders", "Mumbai Indians"],
            "toss_winner": ["Mumbai Indians", "Chennai Super Kings"],
            "toss_decision": ["field", "bat"],
            "winner": ["Mumbai Indians", ""],
            "season": ["2024", "2024"],
            "city": ["Mumbai", "Chennai"],
        }
    )
    df = clean_matches(raw)
    assert len(df) == 1
    assert df.loc[0, "y"] == 1
    assert df.loc[0, "team1_chasing"] == 1
    assert df.loc[0, "home_edge"] == 1


def test_design_shapes():
    raw = pd.DataFrame(
        {
            "team1": ["Mumbai Indians", "Chennai Super Kings"],
            "team2": ["Kolkata Knight Riders", "Mumbai Indians"],
            "toss_winner": ["Mumbai Indians", "Chennai Super Kings"],
            "toss_decision": ["field", "field"],
            "winner": ["Kolkata Knight Riders", "Chennai Super Kings"],
            "season": ["2023", "2024"],
            "city": ["Kolkata", "Chennai"],
        }
    )
    tape = encode(clean_matches(raw))
    dm = design_matrices(tape)
    assert dm["y"].shape == (2,)
    assert dm["n_teams"] == 3
    assert set(dm["y"]).issubset({0, 1})

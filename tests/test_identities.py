"""Synthetic identities the NFL project failed: P(win) comes from the same eta as the mean."""

import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ipl_hier.metrics import brier, log_loss


def test_brier_perfect_is_zero():
    y = np.array([1, 0, 1, 0])
    p = np.array([1.0, 0.0, 1.0, 0.0])
    assert brier(y, p) == 0.0


def test_log_loss_fifty_fifty():
    y = np.array([1, 0])
    p = np.array([0.5, 0.5])
    assert abs(log_loss(y, p) - np.log(2)) < 1e-12


def test_probability_is_monotone_in_strength_gap():
    def p(gap):
        return 1 / (1 + np.exp(-gap))

    assert p(2) > p(0) > p(-2)
    assert abs(p(0) - 0.5) < 1e-12
    assert p(3) > 0.9

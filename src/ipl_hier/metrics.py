import numpy as np


def brier(y, p):
    return float(np.mean((p - y) ** 2))


def log_loss(y, p, eps=1e-12):
    p = np.clip(p, eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

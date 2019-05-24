import numpy as np
import math


def mean_squared_error(x, y):
    assert len(x) == len(y), "calling MSE on vectors of different size"
    r = np.square(x - y)
    return np.sum(r) / len(r)


def trimmed_mean_squared_error(x, y, alpha=0.25):
    assert len(x) == len(
        y), "calling trimmed loss on vectors of different size"
    r = np.square(x - y)
    h = int(math.floor((1 - alpha) * len(x)))
    idx = np.argpartition(r, h)
    tr = r[idx[:h]]
    return np.sum(tr) / h

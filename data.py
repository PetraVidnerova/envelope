import numpy as np


def select(xo, yo, lower, upper):

    lower = lower.reshape(yo.shape)
    upper = upper.reshape(yo.shape)

    sel1 = yo > lower
    sel2 = yo < upper
    selected = sel1 & sel2

    x = xo[selected]
    y = yo[selected]

    return x, y


def load_data(name, lower=None, upper=None):

    matrix = np.loadtxt(name+".txt")
    x = matrix[:, :-1]
    y = matrix[:, -1]

    if lower is not None and upper is not None:

        lower = lower.reshape(y.shape)
        upper = upper.reshape(y.shape)

        sel1 = y > lower
        sel2 = y < upper
        selected = sel1 & sel2

        x = x[selected]
        y = y[selected]

    return x, y

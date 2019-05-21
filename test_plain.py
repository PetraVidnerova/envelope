import sys
from importlib import import_module

import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras import losses

from data import load_data
from utils import mean_squared_error, trimmed_mean_squared_error

# seed(142)
# set_random_seed(142)


def fit_n(n, model_generator, loss, x, y):
    """ Fits n models and returns the one with the lowest cost. """

    models = [model_generator(loss) for _ in range(n)]
    losses = []
    for model in models:
        history = model.fit(x, y,
                            batch_size=32,
                            epochs=2000,
                            verbose=1)
        loss_i = history.history["loss"][-1]
        losses.append(loss_i)

    print(losses)
    winner = models[np.argmin(losses)]
    return winner


if __name__ == "__main__":

    index = sys.argv[1]

    if len(sys.argv) > 3:
        lower = np.load(sys.argv[2])
        upper = np.load(sys.argv[3])
    else:
        lower, upper = None, None

    x1, y1 = load_data("data/autompg")
    x, y = load_data("data/autompg", lower, upper)

    mod = import_module(sys.argv[-1])
    model = mod.model 

    # model
    trained_model = fit_n(5, model, losses.mean_squared_error, x, y)
    ym1 = trained_model.predict(x1)

    if lower is None and upper is None:
        trained_model.save("mlp_autompg_{index}_plain.h5".format(index=index))
        np.save("mlp_autompg_{index}_y_plain".format(index=index), ym1)
    else:
        trained_model.save("mlp_autompg_{index}_restricted_{lower}_{upper}.h5".format(
            index=index, lower=sys.argv[1], upper=sys.argv[2]))
        np.save("mlp_autompg_{index}_y_restricted_{lower}_{upper}".format(
            index=index, lower=sys.argv[1], upper=sys.argv[2]), ym1)

    # calculate final loss
    ym1 = ym1.squeeze()
    print("MSE:  ", mean_squared_error(y1, ym1))
    print("TMSE: ", trimmed_mean_squared_error(y1, ym1))

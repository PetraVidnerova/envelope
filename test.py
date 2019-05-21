import sys
from importlib import import_module 

import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


from data import load_data
from loss import CustomLoss


# seed(42)
# set_random_seed(42)



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

    x, y = load_data("data/autompg")

    index = sys.argv[1]
    tau1 = float(sys.argv[2])
    tau2 = float(sys.argv[3])

    mod = import_module(sys.argv[4])
    model = mod.model 

    # model 1 - upper bound
    loss = CustomLoss(tau1)
    model1 = fit_n(5, model, loss.loss, x, y)
    ym1 = model1.predict(x)
    model1.save("mlp_autompg_{}_{}.h5".format(index, tau1))
    np.save("mlp_autompg_{}_y_{}".format(index, tau1), ym1)

    # model 2 - lower bound
    loss = CustomLoss(tau2)
    model2 = fit_n(5, model, loss.loss, x, y)
    ym2 = model2.predict(x)
    model2.save("mlp_autompg_{}_{}.h5".format(index, tau2))
    np.save("mlp_autompg_{}_y_{}".format(index, tau2), ym2)

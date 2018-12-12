import sys
from random import seed as rseed 
from numpy.random import seed
from tensorflow import set_random_seed

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import RMSprop
from keras.initializers import RandomUniform

from rbflayer import RBFLayer

from data import load_data
from loss import CustomLoss
from test import fit_n


import matplotlib.pyplot as plt


def rbf(loss):

    model = Sequential()
    model.add(RBFLayer(10, input_shape=(1,), initializer=RandomUniform(-1.0, 1.0), betas=1.0))
    model.add(Dense(1, use_bias=False))

    model.summary()

    model.compile(loss=loss,
                  optimizer=RMSprop()
                  )
    return model



    
if __name__ == "__main__":

    rseed(42)
    seed(42)
    set_random_seed(42)
    
    x, y = load_data("data4")

    tau1 = float(sys.argv[1])
    tau2 = float(sys.argv[2])
    
    # model 1 - upper bound 
    loss = CustomLoss(tau1)
    model = fit_n(5, rbf, loss.loss, x, y)
    ym1 = model.predict(x)

    # model 2 - lower bound
    loss = CustomLoss(tau2)
    model = fit_n(5, rbf, loss.loss, x, y)
    ym2 = model.predict(x)
    
    
    fig, ax = plt.subplots()
    ax.scatter(x, y, color='b')
    ax.plot(x, ym1, color='r')
    ax.plot(x, ym2, color='r')

    plt.savefig(f"rbf_{tau1}_{tau2}.png", bbox_inches='tight')

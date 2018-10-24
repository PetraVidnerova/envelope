import sys
from numpy.random import seed
from tensorflow import set_random_seed

import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD


from data import load_data
from loss import CustomLoss

import matplotlib.pyplot as plt


seed(42)
set_random_seed(42)

def mlp(loss):

    model = Sequential()
    model.add(Dense(5, input_shape=(1,), activation="sigmoid"))
    model.add(Dense(1))

    model.summary()

    model.compile(loss=loss,
                  optimizer=SGD()
                  )
    return model

def fit_n(n, model_generator, loss, x, y):
    """ Fits n models and returns the one with the lowest cost. """ 
    
    models = [ model_generator(loss) for _ in range(n) ]
    losses = []
    for model in models:
        history = model.fit(x, y,
                            batch_size=128,
                            epochs=20000,
                            verbose=1)
        loss_i = history.history["loss"][-1]
        losses.append(loss_i)

    print(losses)
    winner = models[np.argmin(losses)]
    return winner

    
if __name__ == "__main__":

    x, y = load_data("data1")

    tau1 = float(sys.argv[1])
    tau2 = float(sys.argv[2])
    
    # model 1 - upper bound 
    loss = CustomLoss(tau1)
    model = fit_n(5, mlp, loss.loss, x, y)
    ym1 = model.predict(x)

    # model 2 - lower bound
    loss = CustomLoss(tau2)
    model = fit_n(5, mlp, loss.loss, x, y)
    ym2 = model.predict(x)
    
    
    fig, ax = plt.subplots()
    ax.scatter(x, y, color='b')
    ax.plot(x, ym1, color='r')
    ax.plot(x, ym2, color='r')

    #plt.show()    
    plt.savefig(f"mlp_{tau1}_{tau2}.png", bbox_inches='tight')

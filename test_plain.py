import sys
from numpy.random import seed
from tensorflow import set_random_seed

import numpy as np

from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.optimizers import SGD, RMSprop, Adam
from keras import losses
from keras import backend as K


from data import load_data
from loss import CustomLoss
from utils import mean_squared_error, trimmed_mean_squared_error

import matplotlib.pyplot as plt


#seed(142)
#set_random_seed(142)

def mlp(loss):

    model = Sequential()
    model.add(Dense(32, input_shape=(3,), activation="sigmoid"))
    model.add(Dense(16, activation="sigmoid"))
#    model.add(Dense(16, activation="sigmoid"))
    model.add(Dense(1))

    model.summary()

    model.compile(loss=loss,
                  optimizer=Adam(),
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
    
    index = sys.argv[1] 

    if len(sys.argv) > 3:
        lower = np.load(sys.argv[2])
        upper = np.load(sys.argv[3])
    else:
        lower, upper = None, None

        
    x1, y1 = load_data("data_turisti_sel")
    x, y = load_data("data_turisti_sel", lower, upper)
    

    # model
    model = fit_n(5, mlp, losses.mean_squared_error, x, y)
    ym1 = model.predict(x1)

    if lower is None and upper is None:
        model.save("mlp{index}_plain.h5".format(index=index))
        np.save("mlp{index}_y_plain".format(index=index), ym1)
    else:
        model.save("mlp{index}_restricted_{lower}_{upper}.h5".format(index=index, lower=sys.argv[1], upper=sys.argv[2]))
        np.save("mlp{index}_y_restricted_{lower}_{upper}".format(index=index, lower=sys.argv[1], upper=sys.argv[2]), ym1)

    # calculate final loss 
    ym1 = ym1.squeeze()
    print("MSE:  ", mean_squared_error(y1, ym1))
    print("TMSE: ", trimmed_mean_squared_error(y1, ym1))


   
   # fig, ax = plt.subplots()
   # ax.scatter(x1, y1, color='b')
   # ax.plot(x1, ym1, color='r')
   #
   # #plt.show()    
   # if lower is None and upper is None:
   #     plt.savefig("mlp_plain.eps", bbox_inches='tight')
   # else:
   #     plt.savefig("mlp_restricted_{}_{}.eps".format(sys.argv[1], sys.argv[2]), bbox_inches='tight')
   

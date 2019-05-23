from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import RMSprop
from rbflayer import RBFLayer, RandomUniform


def model(loss, n_inputs):

    rbf = Sequential()
    rbf.add(RBFLayer(16, input_shape=(n_inputs,),
                     initializer=RandomUniform(-1.0, 1.0), betas=1.0))
    rbf.add(Dense(1, use_bias=False))

    rbf.summary()

    rbf.compile(loss=loss,
                optimizer=RMSprop()
                )
    return rbf

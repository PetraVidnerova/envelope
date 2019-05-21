from keras.models import Sequential
from keras.layers import Dense 
from keras.optimizers import Adam 


def model(loss):

    net = Sequential()
    net.add(Dense(16, input_shape=(4,), activation="tanh"))
    net.add(Dense(8, activation="sigmoid"))
#    net.add(Dense(8, activation="sigmoid"))
    net.add(Dense(1, activation="tanh"))

    net.summary()

    net.compile(loss=loss,
                optimizer=Adam(),
    )
    return net


    

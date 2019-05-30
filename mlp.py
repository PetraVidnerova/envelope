from keras.models import Sequential
from keras.layers import Dense 
from keras.optimizers import Adam 


def model(loss, n_inputs):

    net = Sequential()
    net.add(Dense(16, input_shape=(n_inputs,), activation="sigmoid"))
    net.add(Dense(8, activation="sigmoid"))
#    net.add(Dense(8, activation="sigmoid"))
    net.add(Dense(1))

    net.summary()

    net.compile(loss=loss,
                optimizer=Adam(),
    )
    return net


    

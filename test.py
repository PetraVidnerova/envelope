from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import RMSprop


from data import load_data
from loss import CustomLoss

import matplotlib.pyplot as plt


def mlp(loss):

    model = Sequential()
    model.add(Dense(5, input_shape=(1,), activation="sigmoid"))
    model.add(Dense(1))

    model.summary()

    model.compile(loss=loss,
                  optimizer=RMSprop()
                  )
    return model



    
if __name__ == "__main__":

    x, y = load_data("data1")

    # model 1 - upper bound 
    loss = CustomLoss(0.9, 0.1)
    model = mlp(loss.loss)

    model.fit(x, y,
              batch_size=100,
              epochs=30000,
              verbose=1
              )

    ym1 = model.predict(x)

    # model 2 - lower bound
    loss = CustomLoss(0.1, 0.9)
    model = mlp(loss.loss)

    model.fit(x, y,
              batch_size=100,
              epochs=30000,
              verbose=1
              )
    ym2 = model.predict(x)
    
    
    fig, ax = plt.subplots()
    ax.scatter(x, y, color='b')
    ax.plot(x, ym1, color='r')
    ax.plot(x, ym2, color='r')

    plt.show()    
    #plt.savefig("obalka2.png", bbox_inches='tight')

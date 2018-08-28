from keras import backend as K

class CustomLoss():
    """ Implements Honza Kalina's loss function. """

    def __init__(self, a, b):
        self.a = a
        self.b = b

    def rho(self, x):
        # if x > 0: a*abs(x) else b*abs(x) 
        return K.switch(x > 0,
                        self.a * K.abs(x),
                        self.b * K.abs(x))
    
    def  loss(self, y_true, y_pred):
        diff = y_true - y_pred
        return K.sum(self.rho(diff))

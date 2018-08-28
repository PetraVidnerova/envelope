import numpy as np

def load_data(name):
    matrix = np.loadtxt(name+".txt")
    x = matrix[:,0]
    y = matrix[:,1]
    return x, y
    

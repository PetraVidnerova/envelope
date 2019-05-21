import random
from math import sin
import numpy as np

for x in np.linspace(-1.0, 1.0, 200):
    print(x, sin(10*x)+random.random())

for x in np.linspace(-1.0, 1.0, 30):
    print(x, 2.5)
    
#for i in range(3):
#    x = 0 + 0.1*random.random()
#    y = 2.5 + random.random()
#    print(x, y)

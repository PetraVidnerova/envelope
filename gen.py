import random
from math import sin

for i in range(100):
    x = i  * 0.10 
    print(x, sin(x)+random.random())

    
for i in range(3):
    x = 1.0 + random.random()
    y = 2.5 + random.random()
    print(x, y)

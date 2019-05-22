import sys

import numpy as np 
import matplotlib.pyplot as plt 

from data import load_data 

data_name = sys.argv[1] 
y_name  = sys.argv[2] 
x_index = int(sys.argv[3]) 

x, y = load_data(data_name) 
yp = np.load(y_name) 


fig, ax = plt.subplots() 

print(x.shape, y.shape)
ax.scatter(x[:, x_index], y, color="b") 

print(x.shape, yp.shape) 
ax.scatter(x[:, x_index], yp, color="r") 

plt.savefig("{}_{}_{}.eps".format(data_name, x_index, y_name),
            bbox_inches='tight') 


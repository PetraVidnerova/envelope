import sys

import numpy as np

name = sys.argv[1]

num = 0
labels = eval(sys.argv[2]) 
print(labels)


plain = ["{}_{}_plain.txt".format(name, i) for i in labels]
trimmed1 = ["{}_{}_trimmed_0.1_0.9.txt".format(name, i) for i in labels ]
trimmed2 = ["{}_{}_trimmed_0.2_0.8.txt".format(name, i) for i in labels ]
trimmed3 = ["{}_{}_trimmed_0.3_0.7.txt".format(name, i) for i in labels ]


def get_number(line):
    return float(line.split(":")[1])


def grep_mse_tmse(file_list):
    mse = []
    tmse = []
    for file_name in file_list:
        with open(file_name, "r") as f:
            for line in f:
                if line.startswith("MSE"):
                    mse.append(get_number(line))
                elif line.startswith("TMSE"):
                    tmse.append(get_number(line))
    return mse, tmse


print(" & MSE & MSE (std) & TMSE & TMSE (std) \\\\")
for file_list in plain, trimmed1, trimmed2, trimmed3:
    mse, tmse = grep_mse_tmse(file_list)
    print(" & {} & {} & {} & {} \\\\".format(np.mean(mse), np.std(mse),
                                             np.mean(tmse), np.std(tmse)))

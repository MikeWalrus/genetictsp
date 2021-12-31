#! /usr/bin/python3

import csv
import tsplib95
import sys
import pandas
import matplotlib.pyplot as plt
import tikzplotlib

try:
    x = sys.argv[2]
    y = sys.argv[3]
except IndexError:
    x = "x"
    y = "y"

df = pandas.read_csv(sys.argv[1], names=[x, y])

plt.plot(df[x], df[y])
plt.xlabel(x)
plt.ylabel(y)

try:
    tikzplotlib.save(sys.argv[4])
except IndexError:
    print("pgfplot not saved")

plt.show()

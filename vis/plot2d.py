#! /usr/bin/python3

import csv
import tsplib95
import sys
import pandas
import matplotlib.pyplot as plt

x = sys.argv[1]
y = sys.argv[2]

df = pandas.read_csv(sys.argv[3], names=[x, y])

plt.plot(df[x], df[y])
plt.xlabel(x)
plt.ylabel(y)

plt.show()

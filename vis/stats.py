#! /usr/bin/python3

import csv
import tsplib95
import sys
import pandas
import matplotlib.pyplot as plt

df = pandas.read_csv(sys.argv[1], names=["max", "avg", "min"])

for i in df:
    df[i] = 1 / df[i]

plt.plot(df["max"])
plt.plot(df["avg"])
plt.plot(df["min"])
plt.show()


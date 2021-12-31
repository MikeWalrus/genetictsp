#! /usr/bin/python3

import csv
import tsplib95
import sys
import pandas
import numpy
import matplotlib.pyplot as plt
import tikzplotlib

df = pandas.read_csv(sys.argv[1], names=["max", "avg", "min"])
for i in df:
    df[i] = 1 / df[i]

total_genration = df.shape[0]
df['generation'] = numpy.arange(0, total_genration)

if total_genration > 1000:
    df = df.iloc[::(int(total_genration/1000)), :]

plt.plot(df["generation"],df["max"])
plt.plot(df["generation"],df["avg"])
plt.plot(df["generation"],df["min"] )

try:
    tikzplotlib.save(sys.argv[2])
except IndexError:
    pass
plt.show()


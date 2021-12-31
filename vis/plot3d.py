#! /usr/bin/python3

import csv
import sys
import pandas
import matplotlib.pyplot as plt
import tikzplotlib

try:
    x = sys.argv[2]
    y = sys.argv[3]
    z = sys.argv[4]
except IndexError:
    x = "x"
    y = "y"
    z = "z"

df = pandas.read_csv(sys.argv[1], names=[x, y, z])

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})


surf = ax.plot_trisurf(df[x], df[y], df[z], cmap=plt.cm.Spectral)
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.xlabel(x)
plt.ylabel(y)
ax.set_zlabel(z)

try:
    tikzplotlib.save(sys.argv[5])
except IndexError:
    print("pgfplot not saved")

plt.show()

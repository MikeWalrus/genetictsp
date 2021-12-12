#! /usr/bin/python3
import matplotlib.pyplot as plt
import csv
import tsplib95
import sys

with open(sys.argv[1]) as f:
    reader = csv.reader(f)
    for row in reader:
        route = row

problem = tsplib95.load(sys.argv[2])


xs = []
ys = []

nodes = []

for node in problem.get_nodes():
    (x, y) = problem.node_coords[node]
    nodes.append((x, y))
    xs.append(x)
    ys.append(y)

a = plt.scatter(xs, ys)

route_x = []
route_y = []

for i in route:
    node = nodes[int(i)]
    route_x.append(node[0])
    route_y.append(node[1])
plt.plot(route_x, route_y)

plt.show()

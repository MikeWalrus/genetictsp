#! /usr/bin/python3

import subprocess
import sys
import csv
import itertools
from concurrent import futures
from numpy import arange
from copy import copy

prog = "genetictsp"
tsp = "fl417.tsp"
generation_max = 9000
expected = 80000

def run(args: list[str]):
    cmd = [prog] + args
    cmd = [str(i) for i in cmd]
    p = subprocess.run(cmd, capture_output=True)
    if p.returncode != 0:
        return generation_max
    return int(p.stdout)

def run_get_avg(args: list[str], n: int):
    generation_sum = 0
    reach_max = 0
    for i in range(n):
        result = run(args)
        generation_sum += result
        if result == generation_max:
            reach_max += 1
        else:
            reach_max = 0
        if reach_max >= 3:
            return generation_max
    return generation_sum / n

def run_dict(args: dict[str, str]):
    args_list = []
    for (name, value) in args.items():
        args_list.append(name)
        args_list.append(value)
    return run_get_avg(args_list, 10)

def population(args, writer):
    for population in range(10, 3000, 10):
        args["-p"] = str(population)
        g = run_dict(args)
        writer.writerow([population, g])

def run_with_param_range(executor, writer, param: list[tuple[str, range]], other_args: dict[str, str]):
    args = other_args
    (names, ranges) = zip(*param)
    combinations = itertools.product(*ranges)
    for values in combinations:
        for (name, value) in zip(names, values):
            args[name] = value
        executor.submit(run_and_write, copy(args), writer, values)

def run_and_write(args, writer, values):
    print(args)
    g = run_dict(args)
    writer.writerow([*values, g])

def main():
    args = {"-i": tsp, "-e": expected, "-g": generation_max, "-p": 1000}
    with open(sys.argv[1], "w", buffering=1) as f:
        with futures.ThreadPoolExecutor(max_workers=8) as executor:
            writer = csv.writer(f)
            param = [("-M", arange(0., 1., 0.02)), ("-C", arange(0., 1., 0.02))]
            run_with_param_range(executor, writer, param, args)
            

if __name__ == "__main__":
    main()

#! /usr/bin/python3

import subprocess
import sys
from icecream import ic
import csv

prog = "./target/release/genetictsp"
tsp = "gr17.tsp"
generation_max = 1000
expected = 2500

def run(args: list[str]):
    cmd = [prog] + args
    p = subprocess.run(cmd, capture_output=True)
    if p.returncode != 0:
        return generation_max
    return int(p.stdout)

def run_get_avg(args: list[str], i: int):
    generation_sum = 0
    for i in range(i):
        generation_sum += run(args)
    return generation_sum / i
        

def run_dict(args: dict[str, str]):
    args_list = []
    for (name, value) in args.items():
        args_list.append(name)
        args_list.append(value)
    ic(args_list)
    return run_get_avg(args_list, 100)

def population(args, writer):
    for population in range(10, 10000, 10):
        args["-p"] = str(population)
        g = run_dict(args)
        writer.writerow([population, g])

def main():
    args = {"-i": tsp, "-e": str(expected), "-g": str(generation_max)};
    with open(sys.argv[1], "w") as f:
        writer = csv.writer(f)
        population(args, writer)

if __name__ == "__main__":
    main()

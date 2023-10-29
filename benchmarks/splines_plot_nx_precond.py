# First execute :
# ./benchmarks/ddc_benchmark_splines --benchmark_format=json --benchmark_out=splines_bench.json
# then execute this code will be able to plot results:
# python3 splines_plot.py /path/to/splines_bench.json

import argparse
import matplotlib.pyplot as plt
import json
import numpy as np

parser = argparse.ArgumentParser(description="Plot bytes_per_second from a JSON file.")
parser.add_argument("json_file", help="Path to the JSON file")
args = parser.parse_args()

with open(args.json_file, 'r') as file:
        data = json.load(file);

# Extract the values at the end of "name" and corresponding "bytes_per_second"
nx_values = sorted(set(int(benchmark["name"].split("/")[1]) for benchmark in data["benchmarks"]))
data_groups = {nx: {"preconditionner_max_block_size": [], "bytes_per_second": [], "gpu_mem_occupancy": []} for nx in nx_values}

for benchmark in data["benchmarks"]:
    nx = int(benchmark["name"].split("/")[1])
    preconditionner_max_block_size = int(benchmark["name"].split("/")[5])
    data_groups[nx]["preconditionner_max_block_size"].append(preconditionner_max_block_size)
    data_groups[nx]["bytes_per_second"].append(benchmark["bytes_per_second"])
    data_groups[nx]["gpu_mem_occupancy"].append(benchmark["gpu_mem_occupancy"])

# Plotting the data for each group
plt.figure(figsize=(8, 6))
for nx, group_data in data_groups.items():
    preconditionner_max_block_size = group_data["preconditionner_max_block_size"]
    scaling = [group_data["bytes_per_second"][i] for i in range(len(preconditionner_max_block_size))]
    # scaling = [group_data["bytes_per_second"][i] / preconditionner_max_block_size[i] / (group_data["bytes_per_second"][0] / preconditionner_max_block_size[0]) for i in range(len(preconditionner_max_block_size))]
    plt.plot(preconditionner_max_block_size, scaling, marker='o', markersize=5, label=f'nx={nx}')

# Plotting the data
plt.grid()
# plt.xscale("log")
plt.xlabel("preconditionner_max_block_size")
plt.ylabel("Bandwidth [B/s]")
plt.title("Bandwidth (ny=100000)")
plt.legend()
plt.savefig("bytes_per_sec.png")

plt.figure(figsize=(8, 6))
for nx, group_data in data_groups.items():
    preconditionner_max_block_size = group_data["preconditionner_max_block_size"]
    gpu_mem_overhead = [group_data["gpu_mem_occupancy"][i]-nx*preconditionner_max_block_size[i]*8 for i in range(len(preconditionner_max_block_size))]
    plt.plot(preconditionner_max_block_size, gpu_mem_overhead, marker='o', markersize=5, label=f'nx={nx}')

# Plotting the data
plt.grid()
plt.xscale("log")
plt.xlabel("preconditionner_max_block_size")
plt.ylabel("Memory overhead [B]")
plt.title("Memory occupancy overhead (occupancy - size of processed data)")
plt.legend()
plt.savefig("gpu_mem_occupancy.png")

plt.close();

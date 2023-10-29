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
data_groups = {nx: {"cols_per_par_chunk": [], "bytes_per_second": [], "gpu_mem_occupancy": []} for nx in nx_values}

for benchmark in data["benchmarks"]:
    nx = int(benchmark["name"].split("/")[1])
    cols_per_par_chunk = int(benchmark["name"].split("/")[3])
    data_groups[nx]["cols_per_par_chunk"].append(cols_per_par_chunk)
    data_groups[nx]["bytes_per_second"].append(benchmark["bytes_per_second"])
    data_groups[nx]["gpu_mem_occupancy"].append(benchmark["gpu_mem_occupancy"])

# Plotting the data for each group
plt.figure(figsize=(8, 6))
for nx, group_data in data_groups.items():
    cols_per_par_chunk = group_data["cols_per_par_chunk"]
    scaling = [group_data["bytes_per_second"][i] for i in range(len(cols_per_par_chunk))]
    plt.plot(cols_per_par_chunk, scaling, marker='o', markersize=5, label=f'nx={nx}')

# Plotting the data
plt.grid()
plt.xscale("log")
plt.xlabel("cols_per_par_chunk")
plt.ylabel("Bandwidth [B/s]")
plt.title("Bandwidth (with ny=100000)")
plt.legend()
plt.savefig("bytes_per_sec.png")

plt.figure(figsize=(8, 6))
for nx, group_data in data_groups.items():
    cols_per_par_chunk = group_data["cols_per_par_chunk"]
    gpu_mem_overhead = [group_data["gpu_mem_occupancy"][i]-nx*100000*8 for i in range(len(cols_per_par_chunk))]
    plt.plot(cols_per_par_chunk, gpu_mem_overhead, marker='o', markersize=5, label=f'nx={nx}')

# Plotting the data
plt.grid()
plt.xscale("log")
plt.xlabel("cols_per_par_chunk")
plt.ylabel("Memory overhead [B]")
plt.title("Memory occupancy overhead (occupancy - size of processed data)")
plt.legend()
plt.savefig("gpu_mem_occupancy.png")

plt.close();

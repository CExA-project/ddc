# Copyright (C) The DDC development team, see COPYRIGHT.md file
#
# SPDX-License-Identifier: MIT

# First execute :
# ./benchmarks/ddc_benchmark_splines --benchmark_format=json --benchmark_out=splines_bench.json
# then execute this code will be able to plot results:
# python3 splines_plot.py /path/to/splines_bench.json

import argparse
from operator import itemgetter 
import matplotlib.pyplot as plt
import json
import numpy as np

parser = argparse.ArgumentParser(description="Plot bytes_per_second from a JSON file.")
parser.add_argument("json_file", help="Path to the JSON file")
args = parser.parse_args()

with open(args.json_file, 'r') as file:
        data = json.load(file);

# Extract the values at the end of "name" and corresponding "bytes_per_second"
nx_values = sorted(set(int(benchmark["name"].split("/")[4]) for benchmark in data["benchmarks"]))
data_groups = {"nx": {nx: {"ny": [], "cols_per_chunk": [], "preconditionner_max_block_size": [], "bytes_per_second": [], "gpu_mem_occupancy": []} for nx in nx_values}}

for benchmark in data["benchmarks"]:
    nx = int(benchmark["name"].split("/")[4])
    data_groups["nx"][nx]["ny"].append(int(benchmark["name"].split("/")[5]))
    data_groups["nx"][nx]["cols_per_chunk"].append(int(benchmark["name"].split("/")[6]))
    data_groups["nx"][nx]["preconditionner_max_block_size"].append(int(benchmark["name"].split("/")[7]))
    data_groups["nx"][nx]["bytes_per_second"].append(benchmark["bytes_per_second"])
    data_groups["nx"][nx]["gpu_mem_occupancy"].append(benchmark["gpu_mem_occupancy"])

#
data_dict = [{
"on_gpu": int(benchmark["name"].split("/")[1]),
"nx": int(benchmark["name"].split("/")[4]),
"ny": int(benchmark["name"].split("/")[5]),
"cols_per_chunk": int(benchmark["name"].split("/")[6]),
"preconditionner_max_block_size": int(benchmark["name"].split("/")[7]),
"bytes_per_second": benchmark["bytes_per_second"],
"gpu_mem_occupancy": benchmark["gpu_mem_occupancy"]
} for benchmark in data["benchmarks"]]




plotter = lambda plt, x_name, y_name, data_dict_sorted, filter : plt.plot([item[x_name] for item in data_dict_sorted if filter(item)], [item[y_name] for item in data_dict_sorted if filter(item)], marker='o', markersize=5, label=f'nx={nx}')

########
## ny ##
########

data_dict_sorted = sorted(data_dict, key=itemgetter("nx","ny"))
plt.figure(figsize=(8, 6))
 
for nx in nx_values:
	filter = lambda item : item["nx"]==nx and item["on_gpu"]
	plotter(plt, "ny", "bytes_per_second", data_dict_sorted, filter)

ny_min = min([item["ny"] for item in data_dict_sorted if item["on_gpu"]])
x = np.linspace(ny_min, 20*ny_min)
plt.plot(x, np.mean([item["bytes_per_second"] for item in data_dict_sorted if item["ny"]==ny_min and item["on_gpu"]])/ny_min*x, linestyle='--', color='black', label='perfect scaling')

plt.grid()
plt.xscale("log")
plt.xlabel("ny")
plt.ylabel("Throughput [B/s]")
plt.title("Throughput on "+str.upper(data["context"]["chip"]));
plt.legend()
plt.savefig("throughput_ny.png")

#############
## gpu_mem ##
#############

plt.figure(figsize=(8, 6))

for nx in nx_values:
	filter = lambda item : item["nx"]==nx and item["on_gpu"] and item["ny"]>=8e3
	plt.plot([item["ny"] for item in data_dict_sorted if filter(item)], [(item["gpu_mem_occupancy"]-nx*item["ny"]*8)/(nx*item["ny"]*8)*100 for item in data_dict_sorted if filter(item)], marker='o', markersize=5, label=f'nx={nx}')

plt.grid()
plt.xscale("log")
plt.xlabel("ny")
plt.ylabel("Relative memory overhead [%]")
plt.title("Relative memory occupancy overhead")
plt.legend()
plt.savefig("gpu_mem_occupancy.png")

########################
## cols_per_chunk ##
########################

# Plotting the data for each group
plt.figure(figsize=(8, 6))
for nx, group_data in data_groups["nx"].items():
    cols_per_chunk = group_data["cols_per_chunk"]
    throughput = [group_data["bytes_per_second"][i] for i in range(len(cols_per_chunk))]
    plt.plot(cols_per_chunk, throughput, marker='o', markersize=5, label=f'nx={nx}')

x = [(int)(data["context"]["cols_per_chunk_ref"]), (int)(data["context"]["cols_per_chunk_ref"])*1.001];
plt.plot(x, [0.99*min([min(group_data["bytes_per_second"]) for nx, group_data in data_groups["nx"].items()]), 1.01*max([max(group_data["bytes_per_second"]) for nx, group_data in data_groups["nx"].items()])], linestyle='dotted', color='black', label='reference config')

# Plotting the data
plt.grid()
plt.xscale("log")
plt.xlabel("cols_per_chunk")
plt.ylabel("Throughput [B/s]")
plt.title("Throughput on "+str.upper(data["context"]["chip"])+" (with ny=100000)");
plt.legend()
plt.savefig("throughput_cols.png")

#####################
## preconditionner ##
#####################

# Plotting the data for each group
plt.figure(figsize=(8, 6))
for nx, group_data in data_groups["nx"].items():
    preconditionner_max_block_size = group_data["preconditionner_max_block_size"]
    throughput = [group_data["bytes_per_second"][i] for i in range(len(preconditionner_max_block_size))]
    plt.plot(preconditionner_max_block_size, throughput, marker='o', markersize=5, label=f'nx={nx}')

x = [(int)(data["context"]["preconditionner_max_block_size_ref"]), (int)(data["context"]["preconditionner_max_block_size_ref"])*1.001];
plt.plot(x, [0.99*min([min(group_data["bytes_per_second"]) for nx, group_data in data_groups["nx"].items()]), 1.01*max([max(group_data["bytes_per_second"]) for nx, group_data in data_groups["nx"].items()])], linestyle='dotted', color='black', label='reference config')

# Plotting the data
plt.grid()
plt.xscale("log")
plt.xlabel("preconditionner_max_block_size")
plt.ylabel("Throughput [B/s]")
plt.title("Throughput on "+str.upper(data["context"]["chip"])+" (with ny=100000)");
plt.legend()
plt.savefig("throughput_precond.png")

plt.close();

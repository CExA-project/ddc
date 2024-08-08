"""
First execute :
./benchmarks/ddc_benchmark_splines --benchmark_format=json --benchmark_out=splines_bench_<lib>_<backend>.json
<lib> is ginkgo or lapack and <backend> is omp, cuda or hip.

then execute this code will be able to plot results:
python comparison.py -dirname <dir>
Please make sure you have all combinations of splines_bench_<lib>_<backend>.json files under <dir>.
"""

import argparse
import matplotlib.pyplot as plt
import json
import pathlib
import numpy as np
import re
import copy
from distutils.util import strtobool

def str2num(datum):
    """Convert string to integer, float, or boolean.

    Args:
        datum (str): The string to be converted.

    Returns:
        int, float, bool, or str: The converted value if possible, otherwise the original string.
    """
    try:
        return int(datum)
    except:
        try:
            return float(datum)
        except:
            try:
                return strtobool(datum)
            except:
                return datum

def parse():
    """Parse command line arguments.

    Returns:
        Namespace: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('-dirname',
                        action='store',
                        nargs='?',
                        const=None,
                        default='../data',
                        type=str,
                        choices=None,
                        help='directory of input file',
                        metavar=None
                       )
    
    return parser.parse_args()

def get_details(func_name):
    """Extract details from the function name.

    Args:
        func_name (str): The function name containing details.

    Returns:
        tuple: A tuple containing a boolean indicating non-uniformity and the spline degree.
    """
    matches = re.findall(r'<(.*?)>', func_name)[0].split(',')
    is_nonuniform, spline_degree = matches
    is_nonuniform = str('true') in is_nonuniform
    
    return is_nonuniform, spline_degree

def preprocess(dirname, arch, lib):
    """Preprocess data from the specified JSON file.

    Args:
        dirname (str): Directory name where the JSON file is located.
        arch (str): Architecture name.
        lib (str): Library name.

    Returns:
        dict: A dictionary containing benchmark details.
    """

    # Read json file
    filename = pathlib.Path(dirname) / f'splines_bench_{lib}_{arch}.json'
    with open(filename, 'r') as f:
        data = json.load(f)
        
    benchmarks = data.get('benchmarks')
    
    keys = ['real_time', 'cpu_time', 'bytes_per_second']
    results = {}
    for bench in benchmarks:
        full_name = bench.get('name')

        # ['name', 'nx', 'ny', 'cols_per_chunk', preconditionner_max_block_size, 'min_time', 'real_time']
        list_of_names = full_name.split('/')
        
        name, nx, ny, cols_per_chunk, preconditionner_max_block_size, *_ = list_of_names
        non_uniform, degree_x =get_details(name)
        
        result = {key: bench.get(key) for key in keys}
        result['degree_x'] = str2num(degree_x)
        result['nx'] = str2num(nx)
        result['ny'] = str2num(ny)
        result['cols_per_chunk'] = str2num(cols_per_chunk)
        result['non_uniform'] = str2num(non_uniform)
        result['kernel_name'] = get_kernel_name(kernels, result['degree_x'], result['non_uniform'])
        
        results[full_name] = result

    return results

def to_list(results, key):
    """Convert dictionary values to a list based on the specified key.

    Args:
        results (dict): Dictionary containing benchmark details.
        key (str): The key whose values need to be extracted into a list.

    Returns:
        list: A list of values corresponding to the specified key.
    """

    _list = [result_dict[key] for result_dict in results.values()]
    _list = set(_list) # remove overlap
    _list = sorted(list(_list)) # To ascending order
        
    return _list

if __name__ == '__main__':
    args = parse()

    kernels = {} # key: kernel_name, values: ([degree_x], non_uniform)
    
    kernels['pttrs'] = ([3], 0)
    kernels['pbtrs'] = ([4, 5], 0)
    kernels['gbtrs'] = ([3, 4, 5], 1)
    
    archs = ['omp', 'cuda', 'hip']
    libs = ['lapack', 'ginkgo']
    
    dirname = args.dirname
    
    size = 18
    fontname = 'Times New Roman'
    plt.rc('xtick', labelsize = size)
    plt.rc('ytick', labelsize = size)
    plt.rc('font', family=fontname)
    
    title_font = {'fontname':fontname, 'size':size*1.5, 'color':'black',
                  'verticalalignment':'bottom'} # bottom of vertical alignment for more space
    axis_font = {'fontname':fontname, 'size':size*1.2}
    
    # Compare the performance of the two libraries

    # Plot
    caption_cict = {('omp', 'lapack'): r'$(a)$', 
                    ('cuda', 'lapack'): r'$(b)$',
                    ('hip', 'lapack'): r'$(c)$',
                    ('omp', 'ginkgo'): r'$(d)$', 
                    ('cuda', 'ginkgo'): r'$(e)$',
                    ('hip', 'ginkgo'): r'$(f)$',
                    }
    range_cict = {('omp', 'lapack'): 0.5, 
                  ('cuda', 'lapack'): 10,
                  ('hip', 'lapack'): 10,
                  ('omp', 'ginkgo'): 0.5, 
                  ('cuda', 'ginkgo'): 0.5,
                  ('hip', 'ginkgo'): 0.5,}
    marker_dict = {3: 'o', 4: 'x', 5: '*'}
    ls_dict = {3: 'solid', 4: 'dashed', 5: 'dotted'}
    uniformity_dict = {0: 'uniform', 1: 'non-uniform'}
    color_dict = {0: 'r', 1: 'b'}
    label_dict = {'lapack': 'Kokkos-kernels', 'ginkgo': 'Ginkgo'}
    
    title_dict = {('omp', 'lapack'): 'Kokkos-kernels (Icelake)', 
                    ('cuda', 'lapack'): 'Kokkos-kernels (A100)', 
                    ('hip', 'lapack'): 'Kokkos-kernels (MI250X)', 
                    ('omp', 'ginkgo'): 'Ginkgo (Icelake)',  
                    ('cuda', 'ginkgo'): 'Ginkgo (A100)', 
                    ('hip', 'ginkgo'): 'Ginkgo (MI250X)', 
                    }
    nx_big = 1024
    for arch in archs:
        for lib in libs:
            caption = caption_cict[(arch, lib)]
            ymax = range_cict[(arch, lib)]
            bench_dict = preprocess(dirname=dirname, arch=arch, lib=lib)
            mat_sizes = to_list(bench_dict, 'ny')

            fig, ax = plt.subplots(figsize=(8, 6))
            
            for non_uniform in uniformity_dict.keys():
                label_added = False
                for spline_degree in ls_dict.keys():
		    # GB/s -> GLUPS
                    all_bandwidth = [1.e-9 * bench_vals['bytes_per_second'] / 8.0 for bench_vals in bench_dict.values() \
                                     if bench_vals['nx'] == nx_big and bench_vals['non_uniform'] == non_uniform and bench_vals['degree_x'] == spline_degree]
            
                    label = uniformity_dict[non_uniform]
                    if not label_added:
                        # Plot lines and label
                        ax.plot(mat_sizes, all_bandwidth, color=color_dict[non_uniform], linestyle=ls_dict[spline_degree], marker=marker_dict[spline_degree], markersize=9, label=label)
                        label_added = True
                    else:
                        # Adding marker without label
                        ax.plot(mat_sizes, all_bandwidth, color=color_dict[non_uniform], linestyle=ls_dict[spline_degree], marker=marker_dict[spline_degree], markersize=9)
            ax.set_xscale('log')
            ax.set_xlabel('Batch size', fontsize=size)
            ax.set_ylabel('GLUPS', fontsize=size)
            ax.set_ylim([0, ymax])
            ax.set_title(title_dict[(arch, lib)], fontsize=size)
            ax.legend(loc='upper left', prop={'size':size})
            ax.grid()
            ax.text(0.05, 1.05, caption, horizontalalignment='center',
                verticalalignment='center', transform=ax.transAxes, fontsize=size*1.5)
            fig.savefig(f'{lib}_{arch}.png', bbox_inches='tight')
            plt.close('all')

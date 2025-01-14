#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 09:25:27 2025

@author: hossein
"""




import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import re
import subprocess
import os
import seaborn as sns

CYCLING_STATE =   (1)
G1_ARR_STATE =    (-1)
G0_STATE =        (-2)
DIFF_STATE =      (-3)
APOP_STATE =      (-4)
WIPED_STATE =     (-5)
CA_CELL_TYPE =    (1)
WT_CELL_TYPE =    (0)
NULL_CELL_TYPE =  (-1)


def interpolator_func(sim_data_total, exp_times):
    
    sim_times = sim_data_total[0,:]
    
    interpolated_array = np.zeros((np.shape(sim_data_total)[0], len(exp_times)))
    
    interpolated_array[0,:] = exp_times
    
    avg_values_list = []
    err_values_list = []
    
    for t_c in range(len(exp_times)):
        exp_t = exp_times[t_c]
        t_err = (exp_t/tau_avg)*tau_err
        
        lower_t = exp_t - t_err
        upper_t = exp_t + t_err
        
        for t_sim_c in range(len(sim_times)):
            if sim_times[t_sim_c]>lower_t and sim_times[t_sim_c]<upper_t:
                avg_values_list.append(sim_data_total[1,t_sim_c])
                err_values_list.append(sim_data_total[2,t_sim_c])
        
        interpolated_array[1,t_c] = np.mean(np.array(avg_values_list))
        interpolated_array[2,t_c] = np.mean(np.array(err_values_list))
        
        
    return interpolated_array

def save_dict_to_file(filename, data_dict):
    """
    Saves the keys and values of a dictionary to a file in aligned columns with up to 8 decimal places.
    
    If the file exists, appends the values as a new row.
    If the file does not exist, creates it and writes the keys and initial values in columns.
    
    Parameters:
    - filename: str, name of the file to write to.
    - data_dict: dict, dictionary with keys and float values.
    """
    file_exists = os.path.exists(filename)
    # Define a fixed column width to ensure proper alignment
    column_width = 20
    
    with open(filename, 'a' if file_exists else 'w') as file:
        if not file_exists:
            # Write keys as the first row, formatted with column width
            keys_line = ''.join(f"{str(k):<{column_width}}" for k in data_dict.keys())
            file.write(keys_line + '\n')
        
        # Write current values as a new row, formatted with column width and 8 decimal places
        values_line = ''.join(f"{v:<{column_width}.8f}" if isinstance(v, float) else f"{v:<{column_width}}" for v in data_dict.values())
        file.write(values_line + '\n')

def WT_cost():
    
    cost = dict()
    
    ## population
    cost['pop'] = 0.0
    sim_data_total = np.loadtxt("overal_pp"+"/"+"norm_WT_ov_pl.txt", delimiter=',')
    
    exp_data = np.loadtxt("exp_data"+"/"+"overal_WT_pure.csv", delimiter=',')
    exp_times = exp_data[0,:]
    
    sim_data = interpolator_func(sim_data_total, exp_times)
    sim_data[1,0] = 1.0
    sim_data[2,0] = 1e-8
    
    # this error is calculated logarithmically.
    # t=0 is excluded because it is always exact in both exp and sim
    exp_ln_errors = exp_data[2,1:]/exp_data[1,1:]
    sim_ln_errors = sim_data[2,1:]/sim_data[1,1:]
    
    weights = 1 / (exp_ln_errors**2 + sim_ln_errors**2)
    weights_sum = np.sum(weights)
    weights = weights / weights_sum
    
    err_pop = np.sum( weights * ( (np.log(exp_data[1,1:])-np.log(sim_data[1,1:]))/np.log(exp_data[1,1:]) )**2 )
    cost['pop'] = err_pop
    ## population
    
    ## division time
    div_time_sim_avg_std = np.loadtxt("overal_pp"+"/"+"div_time_avg_std.txt", delimiter=',')
    div_time_avg_WT = div_time_sim_avg_std[0]
    div_time_pref = np.loadtxt("exp_data"+"/"+"div_time_pref.txt", delimiter=',')
    div_time_pref_WT = div_time_pref[0]
    
    err_div_time = ( (div_time_avg_WT - div_time_pref_WT)/div_time_pref_WT) ** 2
    cost['div_time'] = err_div_time
    ## division time
    
    ## initial composition effect
    ## Not applicable for pure WT
    cost['init_compos'] = 0.0
    ## initial composition effect
    
    ## statistics change
    ## Not applicable for pure WT
    cost['stat_change'] = 0.0
    ## statistics change
    
    cost['tot'] = sum(cost.values())
    
    return cost

with open("compos.txt", "r") as compos:
    compos_key = compos.readline()[:-1]
    compos.close()
    
    
N_runs = np.loadtxt('N_runs.csv', delimiter=',', dtype=int)

time_tilde = np.loadtxt("run_1/pp_data/time.txt", delimiter=',', dtype=float)
tau_info = np.loadtxt("../WT/tau.txt", delimiter=',', dtype=float)
tau_avg = tau_info[0]
tau_err = tau_info[1]
time = tau_avg * time_tilde

N_samples = len(time)

try:
    directory = "cost_opt"
    os.makedirs(directory, exist_ok=True)
except:
    pass

if compos_key=='WT':
    cost = WT_cost()
elif compos_key=='C':
    cost = C_cost()
elif compos_key=='mix':
    cost = mix_cost()
    

save_dict_to_file('cost_opt/cost_hist.txt', cost)
    
    

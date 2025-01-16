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

def weighted_LS_fitting(x_lin_fit, y_lin_fit, y_lin_ERR_fit):
    
    # by me
    d = 2.0 # number of dimension
    
    Delta_mat =     np.array([[0., 0.], [0., 0.]])
    intercept_mat = np.array([[0., 0.], [0., 0.]])
    slope_mat =     np.array([[0., 0.], [0., 0.]])
    
    Delta_mat[0,0] = np.sum(1/y_lin_ERR_fit**2)
    Delta_mat[0,1] = np.sum(x_lin_fit/y_lin_ERR_fit**2)
    Delta_mat[1,0] = np.sum(x_lin_fit/y_lin_ERR_fit**2)
    Delta_mat[1,1] = np.sum(x_lin_fit**2/y_lin_ERR_fit**2)
    Delta = np.linalg.det(Delta_mat)
    
    intercept_mat[0,0] = np.sum(y_lin_fit/y_lin_ERR_fit**2)
    intercept_mat[0,1] = np.sum(x_lin_fit/y_lin_ERR_fit**2)
    intercept_mat[1,0] = np.sum(x_lin_fit*y_lin_fit/y_lin_ERR_fit**2)
    intercept_mat[1,1] = np.sum(x_lin_fit**2/y_lin_ERR_fit**2)
    intercept = (1/Delta)*np.linalg.det(intercept_mat)
    
    slope_mat[0,0] = np.sum(1/y_lin_ERR_fit**2)
    slope_mat[0,1] = np.sum(y_lin_fit/y_lin_ERR_fit**2)
    slope_mat[1,0] = np.sum(x_lin_fit/y_lin_ERR_fit**2)
    slope_mat[1,1] = np.sum(x_lin_fit*y_lin_fit/y_lin_ERR_fit**2)
    slope = (1/Delta)*np.linalg.det(slope_mat)
    
    
    intercept_ERR = np.sqrt(  (1/Delta) * np.sum(x_lin_fit**2/y_lin_ERR_fit**2) )
    slope_ERR =     np.sqrt(  (1/Delta) * np.sum(1.0         /y_lin_ERR_fit**2) )
    # by me
    
    return slope, slope_ERR, intercept, intercept_ERR

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
    Saves the keys and values of a dictionary to a file with keys in the first column,
    values in subsequent columns, and column indices at the top for existing columns only.
    
    If the file exists, appends the values as new columns in the same row.
    If the file does not exist, creates it and writes the keys and column indices.
    
    Parameters:
    - filename: str, name of the file to write to.
    - data_dict: dict, dictionary with keys and float values.
    """
    file_exists = os.path.exists(filename)
    column_width = 20
    
    if not file_exists:
        with open(filename, 'w') as file:
            # Write initial column indices
            file.write(' ' * column_width)  # Indent to align with keys
            
            for i in range(1, 2):  # First set of values is the first column
                file.write(f"{i:<{column_width}}")
            file.write('\n')
            
            # Write keys in the second row
            for key in data_dict.keys():
                file.write(f"{key:<{column_width}}\n")

    with open(filename, 'r+') as file:
        lines = file.readlines()
        file.seek(0)
        
        # Count existing columns by counting entries in the first line
        num_existing_columns = len(lines[0].split())
        new_index = num_existing_columns + 1
        
        # Add the new column index
        lines[0] = lines[0].rstrip('\n') + f"{new_index:<{column_width}}\n"
        
        # Append new values under their respective keys
        for i, (key, value) in enumerate(data_dict.items()):
            formatted_value = f"{value:.8f}" if isinstance(value, float) else str(value)
            lines[i + 1] = lines[i + 1].rstrip('\n') + f"{formatted_value:<{column_width}}\n"
        
        file.writelines(lines)

def read_file_to_dict(filename):
    """
    Reads a text file with keys in the first column, column indices on top, and values in subsequent columns.
    Returns a dictionary with keys from the first column and values from the last column.
    
    Parameters:
    - filename: str, name of the file to read.
    
    Returns:
    - dict: dictionary with keys from the first column and values from the last column.
    """
    result_dict = {}
    
    with open(filename, 'r') as file:
        lines = file.readlines()
        
        # Skip the first line with column indices
        for line in lines[1:]:
            columns = line.split()
            key = columns[0]
            value = float(columns[-1]) if '.' in columns[-1] else int(columns[-1])
            result_dict[key] = value
    
    return result_dict

def cost_initialization():
    
    # overal_cost_weights
    overal_cost_weights = read_file_to_dict("../overal_cost_weights.txt")
    pop_weight = overal_cost_weights['pop_weight']
    div_t_weight = overal_cost_weights['div_t_weight']
    init_compos_weight = overal_cost_weights['init_compos_weight']
    stat_weight = overal_cost_weights['stat_weight']
    # overal_cost_weights
    
    
    # cost val
    cost = dict()
    
    cost['pop_WT_pure'] = 0.0
    cost['pop_C_pure'] = 0.0
    cost['pop_WT_mix'] = 0.0
    cost['pop_C_mix'] = 0.0
    
    cost['div_t_WT_pure'] = 0.0
    cost['div_t_C_pure'] = 0.0
    # cost['div_t_WT_mix'] = 0.0
    # cost['div_t_C_mix'] = 0.0
    
    cost['init_compos_WT'] = 0.0
    cost['init_compos_C'] = 0.0
    
    cost['stat_ch_WT_g0'] = 0.0
    cost['stat_ch_WT_g1'] = 0.0
    cost['stat_ch_WT_g2'] = 0.0
    
    cost['tot'] = 0.0
    # cost val
    
    # cost weights
    cost_weight = dict()
    
    cost_weight['pop_WT_pure'] = pop_weight / 4
    cost_weight['pop_C_pure'] = pop_weight / 4
    cost_weight['pop_WT_mix'] = pop_weight / 4
    cost_weight['pop_C_mix'] = pop_weight / 4
    
    cost_weight['div_t_WT_pure'] = div_t_weight / 2
    cost_weight['div_t_C_pure'] = div_t_weight / 2
    # cost_weight['div_t_WT_mix'] = 0.0
    # cost_weight['div_t_C_mix'] = 0.0
    
    cost_weight['init_compos_WT'] = init_compos_weight / 2
    cost_weight['init_compos_C'] = init_compos_weight / 2
    
    cost_weight['stat_ch_WT_g0'] = stat_weight / 3
    cost_weight['stat_ch_WT_g1'] = stat_weight / 3
    cost_weight['stat_ch_WT_g2'] = stat_weight / 3
    # cost weights
    
    return cost, cost_weight

def WT_cost(cost, cost_weight):
    
    ## population
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
    
    cost['pop_WT_pure'] = np.sum( weights * ( (np.log(exp_data[1,1:])-np.log(sim_data[1,1:]))/np.log(exp_data[1,1:]) )**2 ) * cost_weight['pop_WT_pure']
    ## population
    
    ## division time
    err_div_time = ( (div_time_avg_WT - div_time_pref_WT)/div_time_pref_WT) ** 2
    cost['div_t_WT_pure'] = err_div_time * cost_weight['div_t_WT_pure']
    ## division time

    
    cost['tot'] = 0.0
    cost['tot'] = sum(cost.values())
    
    return cost

def C_cost(cost, cost_weight):
    
    ## population
    sim_data_total = np.loadtxt("overal_pp"+"/"+"norm_C_ov_pl.txt", delimiter=',')
    
    exp_data = np.loadtxt("exp_data"+"/"+"overal_C_pure.csv", delimiter=',')
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
    
    cost['pop_C_pure'] = np.sum( weights * ( (np.log(exp_data[1,1:])-np.log(sim_data[1,1:]))/np.log(exp_data[1,1:]) )**2 ) * cost_weight['pop_C_pure']
    ## population
    
    ## division time
    err_div_time = ( (div_time_avg_C - div_time_pref_C)/div_time_pref_C) ** 2
    cost['div_t_C_pure'] = err_div_time * cost_weight['div_t_C_pure']
    ## division time

    
    cost['tot'] = 0.0
    cost['tot'] = sum(cost.values())
    
    return cost

def mix_cost(cost, cost_weight):
    
    ## population WT
    sim_data_total = np.loadtxt("overal_pp"+"/"+"norm_WT_ov_pl.txt", delimiter=',')
    
    exp_data = np.loadtxt("exp_data"+"/"+"WT_bar_mix_overal.csv", delimiter=',')
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
    
    cost['pop_WT_mix'] = np.sum( weights * ( (np.log(exp_data[1,1:])-np.log(sim_data[1,1:]))/np.log(exp_data[1,1:]) )**2 ) * cost_weight['pop_WT_mix']
    ## population WT
    
    ## population C
    sim_data_total = np.loadtxt("overal_pp"+"/"+"norm_C_ov_pl.txt", delimiter=',')
    
    exp_data = np.loadtxt("exp_data"+"/"+"C_bar_mix_overal.csv", delimiter=',')
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
    
    cost['pop_C_mix'] = np.sum( weights * ( (np.log(exp_data[1,1:])-np.log(sim_data[1,1:]))/np.log(exp_data[1,1:]) )**2 ) * cost_weight['pop_C_mix']
    ## population C
    
    
    ## division time
    ## No div_timecost for mixed organoids
    ## division time
    
    # init compos ( WT and C)
    init_compos_fit_sim = np.loadtxt("overal_pp/init_compos_fit.txt", delimiter=',', dtype=float)
    init_compos_fit_exp = np.loadtxt("exp_data/init_compos_fit_exp.txt", delimiter=',', dtype=float)
    
    slope_WT_sim = init_compos_fit_sim[0,0]
    slope_C_sim  = init_compos_fit_sim[1,0]
    
    slope_WT_exp = init_compos_fit_exp[0,0]
    slope_C_exp  = init_compos_fit_exp[1,0]
    
    cost['init_compos_WT'] =  ( ( (slope_WT_sim - slope_WT_exp) / slope_WT_exp )**2 ) * cost_weight['init_compos_WT']
    cost['init_compos_C']  =  ( ( (slope_C_sim  - slope_C_exp ) / slope_C_exp  )**2 ) * cost_weight['init_compos_C']
    # init compos ( WT and C)
    
    
    # stat change WT
    f_g0_pure = np.mean(np.loadtxt("../WT/overal_pp/WT_G0_fractions.txt", delimiter=','))
    f_g1_pure = np.mean(np.loadtxt("../WT/overal_pp/WT_G1_fractions.txt", delimiter=','))
    f_g2_pure = np.mean(np.loadtxt("../WT/overal_pp/WT_SG2M_fractions.txt", delimiter=','))
    
    f_g0_mix = np.mean(np.loadtxt("overal_pp/WT_G0_fractions.txt", delimiter=','))
    f_g1_mix = np.mean(np.loadtxt("overal_pp/WT_G1_fractions.txt", delimiter=','))
    f_g2_mix = np.mean(np.loadtxt("overal_pp/WT_SG2M_fractions.txt", delimiter=','))
    
    cost_g0 = np.exp( +( (f_g0_pure - f_g0_mix) / f_g0_pure ) * np.abs((f_g0_pure - f_g0_mix) / f_g0_pure) )
    cost_g1 = ( (f_g1_pure - f_g1_mix) / f_g1_pure ) ** 2
    cost_g2 = np.exp( -( (f_g2_pure - f_g2_mix) / f_g2_pure ) * np.abs((f_g2_pure - f_g2_mix) / f_g2_pure) )
    
    cost['stat_ch_WT_g0'] = cost_g0 * cost_weight['stat_ch_WT_g0']
    cost['stat_ch_WT_g1'] = cost_g1 * cost_weight['stat_ch_WT_g1']
    cost['stat_ch_WT_g2'] = cost_g2 * cost_weight['stat_ch_WT_g2']
    # stat change WT
    
    cost['tot'] = 0.0
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

# average division time
div_time_sim_avg_std = np.loadtxt("overal_pp"+"/"+"div_time_avg_std.txt", delimiter=',')

div_time_avg_WT = div_time_sim_avg_std[0]
div_time_err_WT = div_time_sim_avg_std[1]
div_time_avg_C  = div_time_sim_avg_std[2]
div_time_err_C  = div_time_sim_avg_std[3]

div_time_pref = np.loadtxt("exp_data"+"/"+"div_time_pref.txt", delimiter=',')
div_time_pref_WT = div_time_pref[0]
div_time_pref_C  = div_time_pref[2]
# average division time

N_samples = len(time)

try:
    directory = "cost_opt"
    os.makedirs(directory, exist_ok=True)
except:
    pass

cost, cost_weight = cost_initialization()

if compos_key=='WT':
    cost = WT_cost(cost, cost_weight)
elif compos_key=='C':
    cost = C_cost(cost, cost_weight)
elif compos_key=='mix':
    cost = mix_cost(cost, cost_weight)
    

save_dict_to_file('cost_opt/cost_hist.txt', cost)
    
    

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 09:22:25 2024

@author: hossein
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Jul 13 19:34:39 2024

@author: Nemat002
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



def parse_array(value):
    # Remove surrounding brackets and extra spaces
    value = value.strip('[]')
    # Handle multi-line arrays (convert ';' to '],[')
    if ';' in value:
        value = value.replace(';', ',')
    # Add surrounding brackets to make it a proper list format
    value = f'[{value}]'
    
    try:
        # Convert to a NumPy array
        return np.array(eval(value))
    except (SyntaxError, NameError) as e:
        print(f"Error parsing array: {e}")
        return None

def read_custom_csv(filename):
    # Initialize dictionary to hold variables
    variables = {}
    
    # Read the file
    with open(filename, 'r') as file:
        lines = file.readlines()
        
        for line in lines:
            # Skip comments or empty lines
            if line.strip() == '' or line.strip().startswith('##'):
                continue
            
            # Split the line into key and value
            key, value = line.split('=', 1)
            key = key.strip()
            value = value.strip()
            
            # Process based on the type of the value
            if value.startswith('[') and value.endswith(']'):
                # This is a list or array
                variables[key] = parse_array(value)
                
            elif re.match(r'^[\d.]+$', value):
                # This is a number (int or float)
                variables[key] = float(value) if '.' in value else int(value)
            
            elif re.match(r'^[\w]+$', value):
                # This is a string or keyword
                variables[key] = value
            
    # Extract variables from the dictionary
    N_UpperLim = variables.get('N_UpperLim', None)
    NTypes = variables.get('NTypes', None)
    typeR0 = variables.get('typeR0', None)
    typeR2PI = variables.get('typeR2PI', None)
    typeTypeEpsilon = variables.get('typeTypeEpsilon', None)
    
    typeGamma = variables.get('typeGamma', None)
    typeTypeGammaCC = variables.get('typeTypeGammaCC', None)
    typeTypeF_rep_max = variables.get('typeTypeF_rep_max', None)
    typeTypeF_abs_max = variables.get('typeTypeF_abs_max', None)
    R_eq_coef = variables.get('R_eq_coef', None)
    R_cut_coef_force = variables.get('R_cut_coef_force', None)
    
    typeFm = variables.get('typeFm', None)
    typeDr = variables.get('typeDr', None)
    
    G1Border = variables.get('G1Border', None)
    
    typeOmega = variables.get('typeOmega', None)
    typeBarrierW = variables.get('typeBarrierW', None)
    typeSigmaPhi = variables.get('typeSigmaPhi', None)
    typeSigmaFit = variables.get('typeSigmaFit', None)
    barrierPeakCoef = variables.get('barrierPeakCoef', None)
    typeFit0 = variables.get('typeFit0', None)
    Fit_Th_Wall = variables.get('Fit_Th_Wall', None)
    Fit_Th_G0 = variables.get('Fit_Th_G0', None)
    Fit_Th_Diff = variables.get('Fit_Th_Diff', None)
    Fit_Th_Apop = variables.get('Fit_Th_Apop', None)
    
    maxTime = variables.get('maxTime', None)
    dt = variables.get('dt', None)
    dt_sample = variables.get('dt_sample', None)
    samplesPerWrite = variables.get('samplesPerWrite', None)
    printingTimeInterval = variables.get('printingTimeInterval', None)
    
    R_cut_coef_game = variables.get('R_cut_coef_game', None)
    tau = variables.get('tau', None)
    typeTypePayOff_mat_real_C = variables.get('typeTypePayOff_mat_real_C', None)
    typeTypePayOff_mat_real_F1 = variables.get('typeTypePayOff_mat_real_F1', None)
    typeTypePayOff_mat_real_F2 = variables.get('typeTypePayOff_mat_real_F2', None)
    typeTypePayOff_mat_imag_C = variables.get('typeTypePayOff_mat_imag_C', None)
    typeTypePayOff_mat_imag_F1 = variables.get('typeTypePayOff_mat_imag_F1', None)
    typeTypePayOff_mat_imag_F2 = variables.get('typeTypePayOff_mat_imag_F2', None)
    
    # typeOmega0 = variables.get('typeOmega0', None)
    # typeOmegaLim = variables.get('typeOmegaLim', None)
    newBornFitKey = variables.get('newBornFitKey', None)
    
    shrinkageRate = variables.get('shrinkageRate', None)
    
    initConfig = variables.get('initConfig', None)
    
    return {
        'N_UpperLim': N_UpperLim,
        'NTypes': NTypes,
        'typeR0': typeR0,
        'typeR2PI': typeR2PI,
        'typeTypeEpsilon': typeTypeEpsilon,
        
        'typeGamma': typeGamma,
        'typeTypeGammaCC': typeTypeGammaCC,
        'typeTypeF_rep_max': typeTypeF_rep_max,
        'typeTypeF_abs_max': typeTypeF_abs_max,
        'R_eq_coef': R_eq_coef,
        'R_cut_coef_force': R_cut_coef_force,
        
        'typeFm': typeFm,
        'typeDr': typeDr,
        
        'G1Border' : G1Border,
        
        'typeOmega' : typeOmega,
        'typeBarrierW' : typeBarrierW,
        'typeSigmaPhi' : typeSigmaPhi,
        'typeSigmaFit' : typeSigmaFit,
        'barrierPeakCoef' : barrierPeakCoef,
        'typeFit0' : typeFit0,
        'Fit_Th_Wall' : Fit_Th_Wall,
        'Fit_Th_G0' : Fit_Th_G0,
        'Fit_Th_Diff' : Fit_Th_Diff,
        'Fit_Th_Apop' : Fit_Th_Apop,
        
        'maxTime': maxTime,
        'dt': dt,
        'dt_sample': dt_sample,
        'samplesPerWrite': samplesPerWrite,
        'printingTimeInterval': printingTimeInterval,
                
        'R_cut_coef_game' : R_cut_coef_game,
        'tau' :tau,
        'typeTypePayOff_mat_real_C' :typeTypePayOff_mat_real_C,
        'typeTypePayOff_mat_real_F1' :typeTypePayOff_mat_real_F1,
        'typeTypePayOff_mat_real_F2' :typeTypePayOff_mat_real_F2,
        'typeTypePayOff_mat_imag_C' :typeTypePayOff_mat_imag_C,
        'typeTypePayOff_mat_imag_F1' :typeTypePayOff_mat_imag_F1,
        'typeTypePayOff_mat_imag_F2' :typeTypePayOff_mat_imag_F2,
        
        'newBornFitKey' : newBornFitKey,
        
        'shrinkageRate' : shrinkageRate,
        
        'initConfig': initConfig
    }

def read_custom_csv_pp(filename):
    # Initialize dictionary to hold variables
    variables = {}
    
    # Read the file
    with open(filename, 'r') as file:
        lines = file.readlines()
        
        for line in lines:
            # Skip comments or empty lines
            if line.strip() == '' or line.strip().startswith('##'):
                continue
            
            # Split the line into key and value
            key, value = line.split('=', 1)
            key = key.strip()
            value = value.strip()
            
            # Process based on the type of the value
            if value.startswith('[') and value.endswith(']'):
                # This is a list or array
                variables[key] = parse_array(value)
                
            elif re.match(r'^[\d.]+$', value):
                # This is a number (int or float)
                variables[key] = float(value) if '.' in value else int(value)
            
            elif re.match(r'^[\w]+$', value):
                # This is a string or keyword
                variables[key] = value
            
    # Extract variables from the dictionary
    frame_plot_switch = variables.get('frame_plot_switch', None)
    
    
    return {
        'frame_plot_switch': frame_plot_switch
            }

def stats_plotter(fileName):
    
    plt.figure()
    
    plt.plot(time, alive_stat, label='tot')
    plt.plot(time, C_alive_stat, label='Ca', color='g')
    plt.plot(time, C_apop_stat, label='Ca_apop', linestyle='dashed')
    plt.plot(time, WT_alive_stat, label='WT', color='m')
    plt.plot(time, WT_cyc_stat, label='WT_cyc', linestyle='dashed')
    plt.plot(time, WT_g1_cyc_stat, label='WT_g1_cyc', linestyle='dashed')
    plt.plot(time, WT_g1_arr_stat, label='WT_g1_arr', linestyle='dashed')
    plt.plot(time, WT_g1_tot_stat, label='WT_g1_tot', linestyle='dashed')
    plt.plot(time, WT_g0_stat, label='WT_g0', linestyle='dashed')
    plt.plot(time, WT_diff_stat, label='WT_diff', linestyle='dashed')
    plt.plot(time, WT_diff_stat+WT_g0_stat, label='WT_g0_diff', linestyle='dashed')
    plt.plot(time, WT_apop_stat, label='WT_apop')
    
    plt.xlabel("time (h)")
    plt.ylabel("Number")
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.yscale("log")
    plt.grid()
    plt.tight_layout()
    plt.savefig(fileName+".PNG", dpi=200)
    # plt.close()
    
    plt.figure()
    
    plt.plot(time, alive_stat/alive_stat[0], label='tot')
    plt.plot(time, C_alive_stat/C_alive_stat[0], label='Ca', color='g')
    plt.plot(time, WT_alive_stat/WT_alive_stat[0], label='WT', color='m')
    # plt.plot(time, WT_cyc_stat/WT_cyc_stat[0], label='WT_cyc', linestyle='dashed')
    # plt.plot(time, WT_g1_cyc_stat/WT_g1_cyc_stat[0], label='WT_g1_cyc', linestyle='dashed')
    # plt.plot(time, WT_g1_arr_stat/WT_g1_arr_stat[0], label='WT_g1_arr', linestyle='dashed')
    # plt.plot(time, WT_g1_tot_stat, label='WT_g1_tot', linestyle='dashed')
    # plt.plot(time, WT_g0_stat, label='WT_g0', linestyle='dashed')
    # plt.plot(time, WT_diff_stat, label='WT_diff', linestyle='dashed')
    # plt.plot(time, WT_apop_stat, label='WT_apop', linestyle='dashed')
    
    plt.xlabel("time (h)")
    plt.ylabel("Normalized Number")
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.yscale("log")
    plt.grid()
    plt.tight_layout()
    plt.savefig(fileName+"_norm.PNG", dpi=200)
    # plt.close()
    
    plt.figure()
    plt.plot(time, WT_g1_tot_stat/WT_alive_stat, label='g1_frac')
    # plt.plot(time, WT_g0_stat/WT_alive_stat, label='g0_frac')
    # plt.plot(time, WT_diff_stat/WT_alive_stat, label='diff_frac')
    plt.plot(time, (WT_g0_stat+WT_diff_stat)/WT_alive_stat, label='g0_diff_frac')
    
    plt.xlabel("time (h)")
    plt.ylabel("fractions")
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    # plt.yscale("log")
    plt.grid()
    plt.tight_layout()
    plt.savefig(fileName+"_fracs.PNG", dpi=200)
    # plt.close()
    
    
    return 0


def overal_plotter(exp_data_load_switch, save_switch):
    

    
    save_array = np.zeros((3, len(time)))
    
    ### absolute number
    list_of_data = [alive_stat_overal, 
                    C_alive_stat_overal, 
                    C_dead_stat_overal, 
                    WT_alive_stat_overal, 
                    WT_dead_stat_overal,
                    WT_cyc_stat_overal, 
                    WT_g0_diff_stat_overal,
                    WT_g1_arr_stat_overal,
                    WT_g1_cyc_stat_overal,
                    WT_g1_tot_stat_overal,
                    WT_sg2m_stat_overal]
    
    labels_of_data = ['tot', 
                      'C', 
                      'C_dead',
                      'WT',
                      'WT_dead',
                      'WT_cyc',
                      'WT_g0_diff',
                      'WT_g1_arr',
                      'WT_g1_cyc',
                      'WT_g1_tot',
                      'WT_sg2m']
    
    plt.figure()
    
    
    for dataC in range(len(list_of_data)):
        data = list_of_data[dataC].copy()
        data_label = labels_of_data[dataC]
        
        x = time
        y = np.mean(data,axis=0)
        y_err = np.std(data,axis=0)
        plt.plot(x, y, label=data_label)
        plt.fill_between(x, y - y_err, y + y_err, alpha=0.3)
        
        save_array = 0.0 * save_array
        save_array[0,:] = x
        save_array[1,:] = y
        save_array[2,:] = y_err
        np.savetxt("overal_pp"+"/"+data_label+"_ov_pl"+".txt", save_array, fmt='%.4f', delimiter=',')
        
        
    
    # if exp_data_load_switch:
    #     # x, y, y_err = exp_data_loader()
    #     exp_array = np.loadtxt("exp_data"+"/"+"overal_WT_pure.csv", delimiter=',')
        
    #     x = exp_array[0,:]
    #     y = exp_array[1,:]
    #     y_err = exp_array[2,:]
        
    #     plt.scatter(x, y, label='exp')
    #     plt.errorbar(x, y, yerr=y_err)
    
    
    plt.xlabel('time(h)')
    plt.ylabel('Number')
    plt.grid()
    plt.yscale("log")
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.tight_layout()
    
    if save_switch:
        plt.savefig('overal_stat_absolute.PNG', dpi=300)
    ### absolute number
    
    
    
    ### normalized
    list_of_data.clear()
    list_of_data = [alive_stat_overal, 
                    C_alive_stat_overal, 
                    WT_alive_stat_overal]
    
    labels_of_data.clear()
    labels_of_data = ['norm_tot', 
                      'norm_C', 
                      'norm_WT']
    
    list_of_colors = ['b', 'g', 'm']
    
    plt.figure()
    
    for dataC in range(len(list_of_data)):
        data = list_of_data[dataC].copy()
        data = data / data[:,[0]]
        data_label = labels_of_data[dataC]
        
        color = list_of_colors[dataC]
        
        x = time
        y = np.mean(data,axis=0)
        y_err = np.std(data,axis=0)
        plt.plot(x, y, label=data_label, color=color)
        plt.fill_between(x, y - y_err, y + y_err, alpha=0.3, color=color)
        
        save_array = 0.0 * save_array
        save_array[0,:] = x
        save_array[1,:] = y
        save_array[2,:] = y_err
        np.savetxt("overal_pp"+"/"+data_label+"_ov_pl"+".txt", save_array, fmt='%.4f', delimiter=',')
    
    if exp_data_load_switch:
        #C_mix
        exp_array = np.loadtxt("exp_data"+"/"+"C_bar_mix_overal.csv", delimiter=',')
        x = exp_array[0,:]
        y = exp_array[1,:]
        y_err = exp_array[2,:]
        plt.scatter(x, y, label='exp_C_mix')
        plt.errorbar(x, y, yerr=y_err, fmt='o')
        
        #WT_mix
        exp_array = np.loadtxt("exp_data"+"/"+"WT_bar_mix_overal.csv", delimiter=',')
        x = exp_array[0,:]
        y = exp_array[1,:]
        y_err = exp_array[2,:]
        plt.scatter(x, y, label='exp_WT_mix')
        plt.errorbar(x, y, yerr=y_err, fmt='o')
        
        #C_pure
        exp_array = np.loadtxt("exp_data"+"/"+"overal_C_pure.csv", delimiter=',')
        x = exp_array[0,:]
        y = exp_array[1,:]
        y_err = exp_array[2,:]
        plt.scatter(x, y, label='exp_C_pure')
        plt.errorbar(x, y, yerr=y_err, fmt='o')
        
        #WT_pure
        exp_array = np.loadtxt("exp_data"+"/"+"overal_WT_pure.csv", delimiter=',')
        x = exp_array[0,:]
        y = exp_array[1,:]
        y_err = exp_array[2,:]
        plt.scatter(x, y, label='exp_WT_pure')
        plt.errorbar(x, y, yerr=y_err, fmt='o')
        
        
    plt.xlabel('time(h)')
    plt.ylabel('Normalized Number')
    plt.grid()
    plt.yscale("log")
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.tight_layout()
    
    if save_switch:
        plt.savefig('overal_stat_norm.PNG', dpi=300)
    ### normalized
    
    
    ### fractions
    list_of_data.clear()
    list_of_data = [WT_g1_tot_stat_overal, 
                    WT_g0_diff_stat_overal,
                    WT_sg2m_stat_overal]
    
    labels_of_data.clear()
    labels_of_data = ['frac_WT_g1', 
                      'frac_WT_g0_diff',
                      'frac_WT_sg2m']
    
    plt.figure()
    
    for dataC in range(len(list_of_data)):
        data = list_of_data[dataC].copy()
        data = data / WT_alive_stat_overal
        data_label = labels_of_data[dataC]
        
        x = time
        y = np.mean(data,axis=0)
        y_err = np.std(data,axis=0)
        plt.plot(x, y, label=data_label)
        plt.fill_between(x, y - y_err, y + y_err, alpha=0.3)
        
        save_array = 0.0 * save_array
        save_array[0,:] = x
        save_array[1,:] = y
        save_array[2,:] = y_err
        np.savetxt("overal_pp"+"/"+data_label+"_ov_pl"+".txt", save_array, fmt='%.4f', delimiter=',')
    
    plt.xlabel('time(h)')
    plt.ylabel('fractions')
    plt.grid()
    # plt.yscale("log")
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.tight_layout()
    
    if save_switch:
        plt.savefig('overal_stat_fracs.PNG', dpi=300)
    ### fractions
    
    
    
    
    
    return

def overal_saver():
    
    try:
        directory = "overal_pp"
        os.makedirs(directory, exist_ok=True)
    except:
        pass
        
    # C_alive_stat_overal[runC, :] = C_alive_stat.copy()
    # C_apop_stat_overal[runC, :] = C_apop_stat.copy()
    
    # WT_alive_stat_overal[runC, :] = WT_alive_stat.copy()
    # WT_apop_stat_overal[runC, :] = WT_apop_stat.copy()
    # WT_cyc_stat_overal[runC, :] = WT_cyc_stat.copy()
    # WT_diff_stat_overal[runC, :] = WT_diff_stat.copy()
    # WT_g0_stat_overal[runC, :] = WT_g0_stat.copy()
    # WT_g1_arr_stat_overal[runC, :] = WT_g1_arr_stat.copy()
    # WT_g1_cyc_stat_overal[runC, :] = WT_g1_cyc_stat.copy()
    # WT_g1_tot_stat_overal[runC, :] = WT_g1_tot_stat.copy()
    
    # WT_g0_diff_stat_overal[runC, :] = WT_g0_stat.copy() + WT_diff_stat.copy()
    
    
    # np.savetxt(directory+"/"+"time.txt", time, fmt='%.4f')
    # np.savetxt(directory+"/"+"alive_stat_overal.txt", alive_stat_overal, fmt='%d')
    # np.savetxt(directory+"/"+"C_alive_stat.txt", C_alive_stat, fmt='%d')
    # np.savetxt(directory+"/"+"C_apop_stat.txt", C_apop_stat, fmt='%d')
    # np.savetxt(directory+"/"+"WT_alive_stat.txt", WT_alive_stat, fmt='%d')
    # np.savetxt(directory+"/"+"WT_cyc_stat.txt", WT_cyc_stat, fmt='%d')
    # np.savetxt(directory+"/"+"WT_g1_cyc_stat.txt", WT_g1_cyc_stat, fmt='%d')
    # np.savetxt(directory+"/"+"WT_g1_arr_stat.txt", WT_g1_arr_stat, fmt='%d')
    # np.savetxt(directory+"/"+"WT_g1_tot_stat.txt", WT_g1_tot_stat, fmt='%d')
    # np.savetxt(directory+"/"+"WT_g0_stat.txt", WT_g0_stat, fmt='%d')
    # np.savetxt(directory+"/"+"WT_diff_stat.txt", WT_diff_stat, fmt='%d')
    # np.savetxt(directory+"/"+"WT_apop_stat.txt", WT_apop_stat, fmt='%d')
    
    return

def distributions_plotter():
    
    WT_enter_fit_mean = []
    WT_exit_fit_mean = []
    
    WT_enter_fit_err = []
    WT_exit_fit_err = []
    
    for time_Win_C in range(N_time_windows):
        WT_enter_fit_mean.append( np.mean(WT_enter_fit_all_runs[time_Win_C], axis=0) )
        WT_exit_fit_mean.append( np.mean(WT_exit_fit_all_runs[time_Win_C], axis=0) )
        
        WT_enter_fit_err.append( np.std(WT_enter_fit_all_runs[time_Win_C], axis=0)/np.sqrt(N_runs-1) )
        WT_exit_fit_err.append( np.std(WT_exit_fit_all_runs[time_Win_C], axis=0)/np.sqrt(N_runs-1) )
    
    for time_Win_C in range(N_time_windows):
        
        fig, ax1 = plt.subplots()

        # Calculate the bin centers from bin edges
        bin_edges = fit_hist_bins_focused
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Plotting the histogram on the left y-axis
        ax1.bar(bin_centers, WT_enter_fit_mean[time_Win_C], width=np.diff(bin_edges), align='center', alpha=0.7, color='blue', label='Histogram')
        ax1.errorbar(bin_centers, WT_enter_fit_mean[time_Win_C], yerr=WT_enter_fit_err[time_Win_C], fmt='o', color='black', capsize=5, label='Error Bars')
        
        # Vertical line for threshold
        ax1.axvline(x=variables['Fit_Th_Wall'], color='gray', linestyle='--', label='Threshold')
        
        # Label for the left y-axis
        ax1.set_ylabel("PDF", color='blue')
        
        # Create a secondary y-axis for the barrier plot
        ax2 = ax1.twinx()
        
        # Generate and plot the barrier curve on the right y-axis
        x_barrier_plot = np.linspace(variables['Fit_Th_Wall'] + 0.01, bin_centers[-1], 1000)
        y_barrier_plot = variables['barrierPeakCoef'] / (x_barrier_plot - variables['Fit_Th_Wall'])
        ax2.plot(x_barrier_plot, y_barrier_plot, color='red', zorder=10, label='V(F)')
        
        # Label for the right y-axis
        ax2.set_ylabel("Barrier Values", color='red')
        
        # Label for the x-axis
        ax1.set_xlabel("Fitness")
        
        # Title of the plot
        title = "AVG WT entering fitness dist (" + str(time_Win_C * 0.25) + "* T_max)"
        plt.title(title)
        
        # Add legends
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')
        
        # Save the plot with a specified file name
        fileName = "AVG_enter_fit_" + str(int(time_Win_C))
        plt.savefig(fileName, dpi=200)
    
        # Close the plot to avoid display
        plt.close()
        
    for time_Win_C in range(N_time_windows):
        
        # Create a new figure and axis
        fig, ax1 = plt.subplots()
        
        # Calculate the bin centers from bin edges
        bin_edges = fit_hist_bins_focused
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Plotting the histogram on the left y-axis
        ax1.bar(bin_centers, WT_exit_fit_mean[time_Win_C], width=np.diff(bin_edges), align='center', alpha=0.7, color='red', label='Histogram')
        ax1.errorbar(bin_centers, WT_exit_fit_mean[time_Win_C], yerr=WT_exit_fit_err[time_Win_C], fmt='o', color='black', capsize=5, label='Error Bars')
        
        # Vertical line for threshold
        ax1.axvline(x=variables['Fit_Th_Wall'], color='gray', linestyle='--', label='Threshold')
        
        # Label for the left y-axis
        ax1.set_ylabel("PDF", color='red')
        
        # Create a secondary y-axis
        ax2 = ax1.twinx()
        
        # Generate and plot the barrier curve on the right y-axis
        x_barrier_plot = np.linspace(variables['Fit_Th_Wall'] + 0.01, bin_centers[-1], 1000)
        y_barrier_plot = variables['barrierPeakCoef'] / (x_barrier_plot - variables['Fit_Th_Wall'])
        ax2.plot(x_barrier_plot, y_barrier_plot, color='red', zorder=10, label='V(F)')
        
        # Label for the right y-axis (optional)
        ax2.set_ylabel("Barrier Values", color='red')
        
        # Label for the x-axis
        ax1.set_xlabel("Fitness")
        
        # Title of the plot
        title = "AVG WT G1-exiting fitness dist (" + str(time_Win_C * 0.25) + "* T_max)"
        plt.title(title)
        
        # Add legends
        ax1.legend(loc='upper left')
        # ax2.legend(loc='upper right')  # Uncomment if adding data to ax2 with a legend
        
        # Save the plot with a specified file name
        fileName = "AVG_exit_fit_" + str(int(time_Win_C))
        plt.savefig(fileName, dpi=200)
        
        # Close the plot to avoid display
        plt.close()
    
    return

def heatmap_dist_plotter():
    
    phi_heatmap_mean = np.mean(phi_dist_3d, axis=0)
    fit_heatmap_mean = np.mean(fit_dist_3d, axis=0)
    
    np.savetxt("overal_pp"+"/"+"phi_heatmap_mean.txt", phi_heatmap_mean, fmt='%1.6f', delimiter=',')
    np.savetxt("overal_pp"+"/"+"fit_heatmap_mean.txt", fit_heatmap_mean, fmt='%1.6f', delimiter=',')

    
    # fitness
    plt.figure()
    X, Y = np.meshgrid(time, fit_hist_bins[0:-1])
    plt.pcolormesh(X, Y, np.log(fit_heatmap_mean), shading='auto', cmap='viridis')
    plt.colorbar(label="log(Intensity)")
    plt.xlabel("time")
    plt.ylabel("fitness")
    title = 'AVG distribution of fitness (WT)'
    plt.title(title)
    file_name = 'AVG_fit_hist_WT.PNG'
    plt.savefig(file_name, dpi=150)
    plt.close()
    
    
    # phi
    plt.figure()
    X, Y = np.meshgrid(time, phi_hist_bins[0:-1])
    plt.pcolormesh(X, Y, np.log(phi_heatmap_mean), shading='auto', cmap='viridis')
    plt.colorbar(label="Intensity")
    plt.xlabel("time")
    plt.ylabel("phi")
    title = 'AVG distribution of phi (WT)'
    plt.title(title)
    file_name = 'AVG_phi_hist_WT.PNG'
    plt.savefig(file_name, dpi=150)
    plt.close()
    
    return

def growth_factor_plotter():
    
    plt.figure()
    
    # Combine data and labels for box plot
    combined_data = [WT_growth_factors, C_growth_factors]
    sns.boxplot(data=combined_data, palette=["purple", "green"], width=0.5, showfliers=False)
    
    # Scatter data points over the box plot for WT and C
    x_positions = np.array([0, 1])
    jitter_amount = 0.08  # Spread for scattered points to avoid overlap
    
    if WT_alive_stat[0]>0:
        # Scatter WT data points in purple
        plt.scatter(np.random.normal(x_positions[0], jitter_amount, size=len(WT_growth_factors)), WT_growth_factors, 
                    color="purple", alpha=0.7, edgecolor="black", label="WT")
    
    if C_alive_stat[0]>0:
        # Scatter C data points in green
        plt.scatter(np.random.normal(x_positions[1], jitter_amount, size=len(C_growth_factors)), C_growth_factors, 
                    color="green", alpha=0.7, edgecolor="black", label="C")
    
    # Adding titles and labels
    title = "growth ratio at t = 60 h"
    plt.title(title)
    plt.xticks(ticks=[0, 1], labels=["WT", "C"])
    plt.xlabel("Type")
    plt.ylabel(r"$N/N_0$")
    plt.grid()
    plt.legend(fontsize=15)
    plt.savefig("growth_box.PNG", dpi=300)
    
    
    if WT_alive_stat[0]>0:
        
        # plt.figure()
        # plt.scatter(WT_init_percent, WT_growth_factors, color="purple")
        # plt.xlabel("starting ratio (% WT at t=0)", fontsize=12)
        # plt.ylabel(r"N/N_0", fontsize=15)
        # plt.title("WT cells in mixed organoids")
        # plt.savefig("growth_vs_perc_WT.PNG", dpi=300)
        plt.figure()
        plt.scatter(WT_init_percent, WT_growth_factors, color="purple")
        # Fit a linear regression line
        slope, intercept = np.polyfit(WT_init_percent, WT_growth_factors, 1)
        trendline = np.poly1d((slope, intercept))
        plt.plot(WT_init_percent, trendline(WT_init_percent), color="purple", linestyle="--", label="Fit Line")
        # Labels and title
        plt.xlabel("Starting ratio (% WT at t=0)", fontsize=20)
        plt.ylabel(r"$N/N_0$", fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        # plt.title("WT cells in mixed organoids")
        # Optional: show legend for the fit line
        # plt.legend()
        # Save the figure
        plt.tight_layout()
        plt.savefig("growth_vs_perc_WT.PNG", dpi=300)
        plt.show()
    
    
    
    # plt.figure()
    # plt.scatter(1-WT_init_percent, C_growth_factors, color='green')
    # plt.xlabel("starting ratio (% C at t=0)", fontsize=12)
    # plt.ylabel(r"N/N_0", fontsize=15)
    # plt.title("C cells in mixed organoids")
    # plt.savefig("growth_vs_perc_C.PNG", dpi=300)
    
    if C_alive_stat[0]>0:
        
        plt.figure()
        plt.scatter(1 - WT_init_percent, C_growth_factors, color='green')
        # Fit a linear regression line
        slope, intercept = np.polyfit(1 - WT_init_percent, C_growth_factors, 1)
        trendline = np.poly1d((slope, intercept))
        plt.plot(1 - WT_init_percent, trendline(1 - WT_init_percent), color="g", linestyle="--", label="Fit Line")
        # Labels and title
        plt.xlabel("Starting ratio (% C at t=0)", fontsize=20)
        plt.ylabel(r"$N/N_0$", fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        # plt.title("C cells in mixed organoids")
        # Optional: show legend for the fit line
        # plt.legend()
        # Save the figure
        plt.tight_layout()
        plt.savefig("growth_vs_perc_C.PNG", dpi=300)
        plt.show()
    
    return

def states_box_plotter():
    
    plt.figure()
    
    # Combine data and labels for box plot
    combined_data = [WT_G1_fractions, WT_SG2M_fractions, WT_G0_fractions]
    sns.boxplot(data=combined_data, palette=["purple", "yellow", "grey"], width=0.5, showfliers=False)
    
    # Scatter data points over the box plot for WT and C
    x_positions = np.array([0, 1, 2])
    jitter_amount = 0.08  # Spread for scattered points to avoid overlap
    
    # Scatter WT data points in purple
    plt.scatter(np.random.normal(x_positions[0], jitter_amount, size=len(WT_G1_fractions)), WT_G1_fractions, 
                color="purple", alpha=0.7, edgecolor="black", label="G1 percentage")
    
    # Scatter C data points in green
    plt.scatter(np.random.normal(x_positions[1], jitter_amount, size=len(WT_SG2M_fractions)), WT_SG2M_fractions, 
                color="yellow", alpha=0.7, edgecolor="black", label="S/G2/M percentage")
    
    plt.scatter(np.random.normal(x_positions[2], jitter_amount, size=len(WT_G0_fractions)), WT_G0_fractions, 
                color="grey", alpha=0.7, edgecolor="black", label="G0 percentage")
    
    # Adding titles and labels
    title = "WT phases distribution at t = 60 h"
    plt.title(title)
    plt.xticks(ticks=[0, 1, 2], labels=["G1", "S/G2/M", "G0"])
    plt.xlabel("Cell state", fontsize=15)
    plt.ylabel("percent", fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.grid()
    plt.legend(fontsize=15)
    plt.tight_layout()
    plt.savefig("states_stats.PNG", dpi=300)
    
    return

def mean_cycle_time_func_overal():
    
    # WT_division_time = []
    # CA_division_time = []
    
    WT_dt = []
    CA_dt = []
    
    for i in range(N_runs):
    
        WT_dt_run = np.loadtxt("run_"+str(i+1)+"/pp_data/WT_div_dt.txt", delimiter=',')
        CA_dt_run = np.loadtxt("run_"+str(i+1)+"/pp_data/CA_div_dt.txt", delimiter=',')
    
    
        WT_dt = WT_dt.copy() + list(WT_dt_run).copy()
        CA_dt = CA_dt.copy() + list(CA_dt_run).copy()
    
    WT = WT_dt.copy()
    CA = CA_dt.copy()
    
    
    
    
    # Calculate mean and standard deviation for WT and CA
    wt_mean, wt_std = np.mean(WT), np.std(WT)
    ca_mean, ca_std = np.mean(CA), np.std(CA)
    
    # Plot histograms
    plt.figure(figsize=(10, 6))
    hist_CA, bins_CA, _ = plt.hist(CA, bins=15, color='green', alpha=0.5, label='CA', edgecolor='black', density=True)
    hist_WT, bins_WT, _ = plt.hist(WT, bins=15, color='purple', alpha=0.7, label='WT', edgecolor='black', density=True)
    
    # Find the maximum heights for both histograms
    max_CA = max(hist_CA)
    max_WT = max(hist_WT)
    maxes_candidates = np.array([max_WT, max_CA])
    
    # Position scatter points and whiskers just above the highest bar
    # wt_max_freq = max(np.histogram(WT, bins=15)[0])
    # ca_max_freq = max(np.histogram(CA, bins=15)[0])
    
    scatter_y_pos = max(maxes_candidates[~np.isnan(maxes_candidates)])*1.1
    scatter_y_pos_c = scatter_y_pos *1.1
    
    # Scatter plot for the means with whiskers on top
    if (not np.isnan(wt_mean)):
        plt.scatter(wt_mean, scatter_y_pos, color='purple', s=100, marker='o', label=f'WT Mean: {wt_mean:.2f}')
        plt.errorbar(wt_mean, scatter_y_pos, xerr=wt_std, fmt='o', color='purple', capsize=5, label=f'WT ±1 STD: {wt_std:.2f}')
    if (not np.isnan(ca_mean)):
        plt.scatter(ca_mean, scatter_y_pos_c, color='green', s=100, marker='o', label=f'CA Mean: {ca_mean:.2f}')
        plt.errorbar(ca_mean, scatter_y_pos_c, xerr=ca_std, fmt='o', color='green', capsize=5, label=f'CA ±1 STD: {ca_std:.2f}')
    
    
    plt.yscale("log")
    
    # Labels and title
    plt.xlabel("division dt", fontsize=15)
    plt.ylabel("PDF of div_dt", fontsize=15)
    plt.title("Division times of WT and CA cells")
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    # Adjust y-axis
    plt.ylim(0, plt.ylim()[1] * 1.1)
    
    # Add legend
    plt.legend(fontsize=15)
    
    # Show grid
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.savefig("divTimesHistOveralLog.PNG", dpi=150)
    
    plt.yscale("linear")
    plt.savefig("divTimesHistOveralLin.PNG", dpi=150)
    
    
    
    return


try:
    directory = "overal_pp"
    os.makedirs(directory, exist_ok=True)
except:
    pass
    
N_runs = np.loadtxt('N_runs.csv', delimiter=',', dtype=int)

time = np.loadtxt("run_1/pp_data/time.txt", delimiter=',', dtype=float)
N_samples = len(time)

## Reading params
filename = 'run_1/params.csv'
variables = read_custom_csv(filename)
## Reading params

alive_stat_overal = np.zeros((N_runs, N_samples), dtype=float)

C_alive_stat_overal = np.zeros((N_runs, N_samples), dtype=float)
C_dead_stat_overal = np.zeros((N_runs, N_samples), dtype=float)

WT_alive_stat_overal = np.zeros((N_runs, N_samples), dtype=float)
# WT_apop_stat_overal = np.zeros((N_runs, N_samples), dtype=float)
WT_cyc_stat_overal = np.zeros((N_runs, N_samples), dtype=float)
WT_diff_stat_overal = np.zeros((N_runs, N_samples), dtype=float)
WT_g0_stat_overal = np.zeros((N_runs, N_samples), dtype=float)
WT_g1_arr_stat_overal = np.zeros((N_runs, N_samples), dtype=float)
WT_g1_cyc_stat_overal = np.zeros((N_runs, N_samples), dtype=float)
WT_g1_tot_stat_overal = np.zeros((N_runs, N_samples), dtype=float)
WT_sg2m_stat_overal = np.zeros((N_runs, N_samples), dtype=float)

WT_g0_diff_stat_overal = np.zeros((N_runs, N_samples), dtype=float)
WT_dead_stat_overal = np.zeros((N_runs, N_samples), dtype=float)



##### ent, ext, hist ############
fit_hist_bins = np.loadtxt("run_1/pp_data/fit_hist_bins.txt", delimiter=',')
fit_hist_bins_focused = np.loadtxt("run_1/pp_data/fit_hist_bins_focused.txt", delimiter=',')
phi_hist_bins = np.loadtxt("run_1/pp_data/phi_hist_bins.txt", delimiter=',')

N_time_windows = 5

WT_enter_fit_all_runs = [np.zeros((N_runs, len(fit_hist_bins)-1)) for _ in range(N_time_windows)]
WT_exit_fit_all_runs = [np.zeros((N_runs, len(fit_hist_bins)-1)) for _ in range(N_time_windows)]
##### ent, ext, hist ############


######## heatmap dists #######
phi_heatmap_run_1 = np.loadtxt("run_1/pp_data/WT_phi_hist_data.txt", delimiter=',')
n_rows , n_cols = np.shape(phi_heatmap_run_1)
phi_dist_3d = np.zeros((N_runs,n_rows, n_cols))

fit_heatmap_run_1 = np.loadtxt("run_1/pp_data/WT_fit_hist_data.txt", delimiter=',')
n_rows , n_cols = np.shape(fit_heatmap_run_1)
fit_dist_3d = np.zeros((N_runs,n_rows, n_cols))
######## heatmap dists #######

growth_factor_time = 60
for tC in range(len(time)):
    if np.abs(time[tC]-growth_factor_time)<(1e-6):
        growth_factor_tC = tC
WT_growth_factors = np.zeros(N_runs)
C_growth_factors = np.zeros(N_runs)
WT_init_percent = np.zeros(N_runs)

WT_G1_fractions =  np.zeros(N_runs)
WT_SG2M_fractions =  np.zeros(N_runs)
WT_G0_fractions =  np.zeros(N_runs)


for runC in range(N_runs):
    folderName = "run_"+str(runC+1)
    
    alive_stat = np.loadtxt(folderName+"/pp_data/alive_stat.txt", delimiter=',', dtype=int)
    
    C_alive_stat = np.loadtxt(folderName+"/pp_data/C_alive_stat.txt", delimiter=',', dtype=int)
    C_dead_stat = np.loadtxt(folderName+"/pp_data/C_dead_stat.txt", delimiter=',', dtype=int)
    
    WT_alive_stat = np.loadtxt(folderName+"/pp_data/WT_alive_stat.txt", delimiter=',', dtype=int)
    WT_dead_stat = np.loadtxt(folderName+"/pp_data/WT_dead_stat.txt", delimiter=',', dtype=int)
    WT_cyc_stat = np.loadtxt(folderName+"/pp_data/WT_cyc_stat.txt", delimiter=',', dtype=int)
    WT_diff_stat = np.loadtxt(folderName+"/pp_data/WT_diff_stat.txt", delimiter=',', dtype=int)
    WT_g0_stat = np.loadtxt(folderName+"/pp_data/WT_g0_stat.txt", delimiter=',', dtype=int)
    WT_g1_arr_stat = np.loadtxt(folderName+"/pp_data/WT_g1_arr_stat.txt", delimiter=',', dtype=int)
    WT_g1_cyc_stat = np.loadtxt(folderName+"/pp_data/WT_g1_cyc_stat.txt", delimiter=',', dtype=int)
    WT_g1_tot_stat = np.loadtxt(folderName+"/pp_data/WT_g1_tot_stat.txt", delimiter=',', dtype=int)
    WT_sg2m_stat = np.loadtxt(folderName+"/pp_data/WT_sg2m_stat.txt", delimiter=',', dtype=int)
    
    
    WT_growth_factors[runC] = WT_alive_stat[growth_factor_tC]/WT_alive_stat[0]
    C_growth_factors[runC] = C_alive_stat[growth_factor_tC]/C_alive_stat[0]
    WT_init_percent[runC] = WT_alive_stat[0] / alive_stat[0]
    
    WT_G1_fractions[runC] = WT_g1_tot_stat[growth_factor_tC] / WT_alive_stat[growth_factor_tC]
    WT_G0_fractions[runC] = WT_g0_stat[growth_factor_tC] / WT_alive_stat[growth_factor_tC]
    WT_SG2M_fractions[runC] = 1 - WT_G1_fractions[runC] - WT_G0_fractions[runC]
    
    alive_stat_overal[runC, :] = alive_stat.copy()
    
    C_alive_stat_overal[runC, :] = C_alive_stat.copy()
    C_dead_stat_overal[runC, :] = C_dead_stat.copy()
    
    WT_alive_stat_overal[runC, :] = WT_alive_stat.copy()
    WT_dead_stat_overal[runC, :] = WT_dead_stat.copy()
    WT_cyc_stat_overal[runC, :] = WT_cyc_stat.copy()
    WT_diff_stat_overal[runC, :] = WT_diff_stat.copy()
    WT_g0_stat_overal[runC, :] = WT_g0_stat.copy()
    WT_g1_arr_stat_overal[runC, :] = WT_g1_arr_stat.copy()
    WT_g1_cyc_stat_overal[runC, :] = WT_g1_cyc_stat.copy()
    WT_g1_tot_stat_overal[runC, :] = WT_g1_tot_stat.copy()
    WT_sg2m_stat_overal[runC, :] = WT_sg2m_stat.copy()
    
    WT_g0_diff_stat_overal[runC, :] = WT_g0_stat.copy() + WT_diff_stat.copy()
    
    
    ## distributions ##
    for time_Win_C in range(N_time_windows):
        WT_enter_fit_all_runs[time_Win_C][runC,:] = (np.loadtxt(folderName+"/pp_data/enter_fit_hist_plot_"+str(time_Win_C)+".txt", delimiter=',')).copy()
        WT_exit_fit_all_runs[time_Win_C][runC,:] = (np.loadtxt(folderName+"/pp_data/exit_fit_hist_plot_"+str(time_Win_C)+".txt", delimiter=',')).copy()
    ## distributions ##
    
    ## heatmap dists ##
    fit_dist_3d[runC, :,:] = np.loadtxt(folderName+"/pp_data/WT_fit_hist_data.txt", delimiter=',')
    phi_dist_3d[runC, :,:] = np.loadtxt(folderName+"/pp_data/WT_phi_hist_data.txt", delimiter=',')
    ## heatmap dists ##


np.savetxt("overal_pp"+"/"+"WT_growth_factors.txt", WT_growth_factors, fmt='%.4f', delimiter=',')
np.savetxt("overal_pp"+"/"+"C_growth_factors.txt", C_growth_factors, fmt='%.4f', delimiter=',')
np.savetxt("overal_pp"+"/"+"WT_init_percent.txt", WT_init_percent, fmt='%.4f', delimiter=',')
np.savetxt("overal_pp"+"/"+"growth_factor_time.txt", [growth_factor_time], fmt='%.4f', delimiter=',')

np.savetxt("overal_pp"+"/"+"WT_G1_fractions.txt", WT_G1_fractions, fmt='%.4f', delimiter=',')
np.savetxt("overal_pp"+"/"+"WT_G0_fractions.txt", WT_G0_fractions, fmt='%.4f', delimiter=',')
np.savetxt("overal_pp"+"/"+"WT_SG2M_fractions.txt", WT_SG2M_fractions, fmt='%.4f', delimiter=',')

exp_data_load_switch = 1
save_switch = 1

overal_plotter(exp_data_load_switch, save_switch)

growth_factor_plotter()

states_box_plotter()

distributions_plotter()
heatmap_dist_plotter()

mean_cycle_time_func_overal()

# ## overal save
# overal_saver()
# ## overal save

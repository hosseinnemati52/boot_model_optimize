#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 09:11:46 2024

@author: hossein
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 17:09:51 2024

@author: hossein
"""

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
import sys

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
    
    ent_ext_switch = variables.get('ent_ext_switch', None)
    
    
    return {
        'frame_plot_switch': frame_plot_switch,
        'ent_ext_switch' : ent_ext_switch
            }

def stats_plotter(fileName):
    
    plt.figure()
    
    plt.plot(time, alive_stat, label='tot')
    plt.plot(time, C_alive_stat, label='Ca', color='g')
    plt.plot(time, C_dead_stat, label='Ca_dead', linestyle='dashed')
    plt.plot(time, WT_alive_stat, label='WT', color='m')
    plt.plot(time, WT_cyc_stat, label='WT_cyc', linestyle='dashed')
    plt.plot(time, WT_g1_cyc_stat, label='WT_g1_cyc', linestyle='dashed')
    plt.plot(time, WT_g1_arr_stat, label='WT_g1_arr', linestyle='dashed')
    plt.plot(time, WT_g1_tot_stat, label='WT_g1_tot', linestyle='dashed')
    plt.plot(time, WT_g0_stat, label='WT_g0', linestyle='dashed')
    plt.plot(time, WT_diff_stat, label='WT_diff', linestyle='dashed')
    plt.plot(time, WT_diff_stat+WT_g0_stat, label='WT_g0_diff', linestyle='dashed')
    plt.plot(time, WT_dead_stat, label='WT_dead')
    plt.plot(time, WT_sg2m_stat, label='WT_sg2m', linestyle='dashed')
    
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
    plt.plot(time, (WT_sg2m_stat)/WT_alive_stat, label='sg2m_frac')
    
    plt.xlabel("time (h)")
    plt.ylabel("fractions")
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    # plt.yscale("log")
    plt.grid()
    plt.tight_layout()
    plt.savefig(fileName+"_fracs.PNG", dpi=200)
    # plt.close()
    
    
    return 0

def plotter(t, snapshotInd):
    # size_vec = np.zeros(N_sph_tot)
    # scale = 4

    # norm = mcolors.Normalize(vmin=cell_sph.min(), vmax=cell_sph.max())
    # cmap = cm.viridis
    
    # fig, ax = plt.subplots()
    
    fig, (ax1, ax2) = plt.subplots(1, 2)
    
    WT_indices = (cellType==WT_CELL_TYPE) & (cellState > WIPED_STATE)
    C_indices = (cellType==CA_CELL_TYPE) & (cellState > WIPED_STATE)
    
    thresh = 1e-8
    
    if len(WT_indices[WT_indices==1])>0:
        WT_fitness_max = np.max(cellFitness[WT_indices, 0])
        WT_fitness_min = np.min(cellFitness[WT_indices, 0])
        
        if abs(WT_fitness_max-WT_fitness_min)<thresh:
            WT_fitness_max = WT_fitness_min + thresh
    else:
        WT_fitness_max = 0
        WT_fitness_min = 0
    
    
    
    if len(C_indices[C_indices==1])>0:
        C_fitness_max = np.max(cellFitness[C_indices, 0])
        C_fitness_min = np.min(cellFitness[C_indices, 0])
        
        if abs(C_fitness_max-C_fitness_min)<thresh:
            C_fitness_max = C_fitness_min + thresh
    else:
        C_fitness_max = 0
        C_fitness_min = 0
    
    
    normWT = mcolors.Normalize(vmin = WT_fitness_min , vmax = WT_fitness_max)
    normC  = mcolors.Normalize(vmin =  C_fitness_min , vmax =  C_fitness_max)

    for i in range(NCells):
        # # size_vec[i] = scale * r_spheres[type_sph[i]]
        # circle = patches.Circle((cellX[i], cellY[i]), radius=cellR[i], edgecolor='k', facecolor='g'*(cellType[i]) + 'violet'*(1-cellType[i]), alpha=0.8)
        # ax1.add_patch(circle)
        
        if cellState[i]==WIPED_STATE:
            continue
        
        
        
        
        if cellType[i] == CA_CELL_TYPE:
            
            if cellState[i]>APOP_STATE:
                # normalized_fitness = normC(cellFitness[i][0])
                normalized_fitness = normC(0.5* ( cellFitness[i][0] + C_fitness_max))
                color = cm.Greens(normalized_fitness)
            else:
                color = 'k'
        else:
            
            if cellState[i]>APOP_STATE:
                # normalized_fitness = normWT(cellFitness[i][0])
                normalized_fitness = normWT(0.5* ( cellFitness[i][0] +  WT_fitness_max))
                color = cm.Purples(normalized_fitness) 
            else:
                color = 'y'
            
        
        if cellState[i]==DIFF_STATE or cellState[i]==G0_STATE or cellState[i]==APOP_STATE:
            polgon = patches.RegularPolygon((cellX[i], cellY[i]),numVertices=5, radius=cellR[i], edgecolor='k', facecolor=color, alpha=0.8)
            ax1.add_patch(polgon)
        else:
            circle = patches.Circle((cellX[i], cellY[i]), radius=cellR[i], edgecolor='k', facecolor=color, alpha=0.8)
            ax1.add_patch(circle)
    # plt.scatter(x_sph, y_sph, s=size_vec, c=cell_sph, alpha=1, cmap='viridis')
    ax1.axis("equal")
    # plt.xlim((0,Lx))
    # plt.ylim((0,Ly))
    ax1.set_aspect('equal', adjustable='box')
    
    # sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    # sm.set_array(cell_sph)
    # cbar = plt.colorbar(sm, ax=ax)
    # cbar.set_label('Cell Value')

    ax1.set_xlabel('X-axis')
    ax1.set_ylabel('Y-axis')
    
    title = 't = {:.3f}'.format(t)
    ax1.set_title(title)
    # ax1.set_title('Colored Scatter Plot with Circles')
    # plt.colorbar()  # Show color scale
    
    file_name = 'frames/frame_'+str(int(snapshotInd))+'.PNG'
    # ax1.title(title)
    # ax1.grid()
    
    ax2.plot(time, WT_alive_stat[:len(time)]/WT_alive_stat[0], label='WT(t)/WT(0)', color='violet')
    ax2.plot(time, C_alive_stat[:len(time)]/C_alive_stat[0], label='C(t)/C(0)', color='g')
    ax2.plot(time, C_alive_stat[:len(time)]/alive_stat[:len(time)], label='C(t)/tot(t)', color='g', linestyle='dashed')
    ax2.plot(time, WT_alive_stat[:len(time)]/alive_stat[:len(time)], label='WT(t)/tot(t)', color='violet', linestyle='dashed')
    ax2.set_yscale("log")
    
    ax2.legend()
    ax2.grid()
    ax2.set_xlabel('time')
    
    
    # ax1.set_ylabel('Y-axis')
    
    plt.savefig(file_name, dpi=150)
    # plt.show()
    plt.close()
    
    return 0

def landscape_plotter(t, snapshotInd):
    # size_vec = np.zeros(N_sph_tot)
    # scale = 4

    # norm = mcolors.Normalize(vmin=cell_sph.min(), vmax=cell_sph.max())
    # cmap = cm.viridis
    
    # fig, ax = plt.subplots()
    
    normalize_factor = 0.01
    
    fig, (ax1) = plt.subplots(1, 1)
    
    
    
    WT_indices = (cellType==WT_CELL_TYPE) & (cellState > APOP_STATE)
    C_indices = (cellType==CA_CELL_TYPE) & (cellState > APOP_STATE)
    
    thresh = 1e-8
    
    if len(WT_indices[WT_indices==1])>0:
        WT_fitness_max = np.max(cellFitness[WT_indices, 0])
        WT_fitness_min = np.min(cellFitness[WT_indices, 0])
        
        if abs(WT_fitness_max-WT_fitness_min)<thresh:
            WT_fitness_max = WT_fitness_min + thresh
    else:
        WT_fitness_max = 0
        WT_fitness_min = 0
    
    
    
    if len(C_indices[C_indices==1])>0:
        C_fitness_max = np.max(cellFitness[C_indices, 0])
        C_fitness_min = np.min(cellFitness[C_indices, 0])
        
        if abs(C_fitness_max-C_fitness_min)<thresh:
            C_fitness_max = C_fitness_min + thresh
    else:
        C_fitness_max = 0
        C_fitness_min = 0
    
    
    normWT = mcolors.Normalize(vmin = WT_fitness_min , vmax = WT_fitness_max)
    normC  = mcolors.Normalize(vmin =  C_fitness_min , vmax =  C_fitness_max)

    for i in range(NCells):
        # # size_vec[i] = scale * r_spheres[type_sph[i]]
        # circle = patches.Circle((cellX[i], cellY[i]), radius=cellR[i], edgecolor='k', facecolor='g'*(cellType[i]) + 'violet'*(1-cellType[i]), alpha=0.8)
        # ax1.add_patch(circle)
        
        if cellState[i] < APOP_STATE+0.01:
            continue
        
        
        if cellType[i] == 1:
            # normalized_fitness = normC(cellFitness[i][0])
            normalized_fitness = normC(0.5* ( cellFitness[i][0] + C_fitness_max))
            color = cm.Greens(normalized_fitness) 
        else:
            # normalized_fitness = normWT(cellFitness[i][0])
            normalized_fitness = normWT(0.5* ( cellFitness[i][0] +  WT_fitness_max))
            color = cm.Purples(normalized_fitness) 
        
        if cellState[i]==DIFF_STATE or cellState[i]==G0_STATE:
            # polgon = patches.RegularPolygon((cellPhi[i], cellFitness[i,0]),numVertices=5, radius=normalize_factor*cellR[i], edgecolor='k', facecolor=color, alpha=0.8)
            # ax1.add_patch(polgon)
            plt.scatter(cellPhi[i], cellFitness[i,0], alpha=0.8, color = color, marker = "p", zorder=10)
        else:
            # circle = patches.Circle((cellPhi[i], cellFitness[i,0]), radius=normalize_factor*cellR[i], edgecolor='k', facecolor=color, alpha=0.8)
            # ax1.add_patch(circle)
            plt.scatter(cellPhi[i], cellFitness[i,0], alpha=0.8, color = color, marker = "o", zorder=10)
    # plt.scatter(x_sph, y_sph, s=size_vec, c=cell_sph, alpha=1, cmap='viridis')
    # ax1.axis("equal")
    # plt.xlim((0,Lx))
    # plt.ylim((0,Ly))
    # ax1.set_aspect('equal', adjustable='box')
    
    x_min = 0
    x_max = 2*np.pi
    
    y_min = variables['Fit_Th_Apop']
    y_max = 1.5*variables['typeFit0'][1]
    
    x_G1_Border = 2*np.pi * variables['G1Border']
    rect_width = variables['typeBarrierW'][0]
    rect_height = variables['Fit_Th_Wall']-y_min
    LB_of_rect_x = x_G1_Border-0.5*rect_width
    LB_of_rect_y = y_min
    
    plt.plot([x_min, x_G1_Border], [y_min, y_min], linestyle='dashed', color='k')
    plt.plot([x_min, x_min], [y_min, y_max], linestyle='dashed', color='k')
    plt.plot([x_max, x_max], [y_min, y_max], linestyle='dashed', color='k')
    plt.plot([x_G1_Border, x_G1_Border], [y_min, y_max], linestyle='dashed', color='k')
    rect = patches.Rectangle((LB_of_rect_x, LB_of_rect_y), rect_width, rect_height, linewidth=0.01, edgecolor='orange', facecolor='orange', alpha=0.4, zorder=1)
    plt.gca().add_patch(rect)

    plt.plot([x_min, x_max], [variables['Fit_Th_Wall'],variables['Fit_Th_Wall']], linestyle='dashed', color='k')
    plt.plot([x_min, x_max], [variables['Fit_Th_G0'],variables['Fit_Th_G0']], linestyle='dashed', color='k')
    # plt.plot([x_G1_Border, x_G1_Border], [y_min,variables['Fit_Th_G1_arr']], linestyle='dashed', color='k')
    # plt.plot([x_min, x_G1_Border], [variables['Fit_Th_G0'],variables['Fit_Th_G0']], linestyle='dashed', color='k')
    plt.plot([x_min, x_max], [variables['Fit_Th_Diff'],variables['Fit_Th_Diff']], linestyle='dashed', color='k')
    plt.plot([x_min, x_max], [variables['Fit_Th_Apop'],variables['Fit_Th_Apop']], linestyle='dashed', color='k')
    
    
    margin_x = 0.1
    margin_y = 0.1
    
    ax1.set_xlim((x_min-margin_x, x_max+margin_x))
    ax1.set_ylim((y_min-margin_y, y_max+5*margin_y))
    
    # sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    # sm.set_array(cell_sph)
    # cbar = plt.colorbar(sm, ax=ax)
    # cbar.set_label('Cell Value')

    ax1.set_xlabel(r'$\phi$')
    ax1.set_ylabel('Fitness')
    
    title = 't = {:.3f}'.format(t)
    ax1.set_title(title)
    # ax1.set_title('Colored Scatter Plot with Circles')
    # plt.colorbar()  # Show color scale
    
    # file_name = 'frames/frame_'+str(int(snapshotInd))+'.PNG'
    file_name = 'fit_land/frame_'+str(int(snapshotInd))+'.PNG'
    # ax1.title(title)
    # ax1.grid()
    
    # ax2.plot(time, WT_alive_stat[:len(time)]/WT_alive_stat[0], label='WT(t)/WT(0)', color='violet')
    # ax2.plot(time, C_alive_stat[:len(time)]/C_alive_stat[0], label='C(t)/C(0)', color='g')
    # ax2.plot(time, C_alive_stat[:len(time)]/alive_stat[:len(time)], label='C(t)/tot(t)', color='g', linestyle='dashed')
    # ax2.plot(time, WT_alive_stat[:len(time)]/alive_stat[:len(time)], label='WT(t)/tot(t)', color='violet', linestyle='dashed')
    # ax2.set_yscale("log")
    
    # ax2.legend()
    # ax2.grid()
    # ax2.set_xlabel('time')
    
    
    # ax1.set_ylabel('Y-axis')
    
    plt.savefig(file_name, dpi=150)
    # plt.show()
    plt.close()
    
    return 0

def hist_plotter():
    
    plt.figure()
    X, Y = np.meshgrid(time, fit_hist_bins[0:-1])
    plt.pcolormesh(X, Y, WT_fit_hist_data, shading='auto', cmap='viridis')
    plt.colorbar(label="Intensity")
    plt.xlabel("time")
    plt.ylabel("fitness")
    title = 'distribution of fitness (WT)'
    plt.title(title)
    file_name = 'fit_hist_WT.PNG'
    plt.savefig(file_name, dpi=150)
    plt.close()
    
    plt.figure()
    X, Y = np.meshgrid(time, phi_hist_bins[0:-1])
    plt.pcolormesh(X, Y, WT_phi_hist_data, shading='auto', cmap='viridis')
    plt.colorbar(label="Intensity")
    plt.xlabel("time")
    plt.ylabel("Phi")
    title = 'distribution of phi (WT)'
    plt.title(title)
    
    file_name = 'phi_hist_WT.PNG'
    plt.savefig(file_name, dpi=150)
    plt.close()
    
    # title = 'distribution of fitness (C)'
    # file_name = 'fit_hist_C.PNG'
    # plt.savefig(file_name, dpi=150)
    # plt.close()
    
    # title = 'distribution of phi (WT)'
    # file_name = 'phi_hist_WT.PNG'
    # plt.savefig(file_name, dpi=150)
    # plt.close()
    
    # title = 'distribution of phi (C)'
    # file_name = 'phi_hist_C.PNG'
    # plt.savefig(file_name, dpi=150)
    # plt.close()
    
    return

def ent_ext_fitness_dist_saver():
    
    for t_window_C in range(len(time_window_list)):
        np.savetxt("pp_data"+"/"+"enter_fit_data_"+str(int(t_window_C))+".txt", enter_fit_data[t_window_C], fmt='%1.4f')
        np.savetxt("pp_data"+"/"+"exit_fit_data_"+str(int(t_window_C))+".txt", exit_fit_data[t_window_C], fmt='%1.4f')
        
    
    
    
    for t_window_C in range(len(time_window_list)):
        hist_plot, bin_edges = np.histogram(enter_fit_data[t_window_C], bins = fit_hist_bins_focused, density=True)
        np.savetxt("pp_data"+"/"+"enter_fit_hist_plot_"+str(int(t_window_C))+".txt", hist_plot, fmt='%1.4f')
        
    
        plt.figure()
        # Calculate the bin centers from bin edges
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
       
        # Plotting the histogram
        plt.bar(bin_centers, hist_plot, width=np.diff(bin_edges), align='center', alpha=0.7, color='blue')
        plt.xlabel("fitness")
        plt.ylabel("PDF")
        title = "entering fitness dist " + "("+str(t_window_C*0.25)+"* T_max)"
        plt.title(title)
        fileName = "enter_fit_"+str(int(t_window_C))
        plt.savefig("extra_plot/"+fileName, dpi=200)
        plt.close()
        
    for t_window_C in range(len(time_window_list)):
        hist_plot, bin_edges = np.histogram(exit_fit_data[t_window_C], bins = fit_hist_bins_focused, density=True)
        np.savetxt("pp_data"+"/"+"exit_fit_hist_plot_"+str(int(t_window_C))+".txt", hist_plot, fmt='%1.4f')
        
    
        plt.figure()
        # Calculate the bin centers from bin edges
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
       
        # Plotting the histogram
        plt.bar(bin_centers, hist_plot, width=np.diff(bin_edges), align='center', alpha=0.7, color='red')
        plt.xlabel("fitness")
        plt.ylabel("PDF")
        title = "G1-exiting fitness dist " + "("+str(t_window_C*0.25)+"* T_max)"
        plt.title(title)
        fileName = "exit_fit_"+str(int(t_window_C))
        plt.savefig("extra_plot/"+fileName, dpi=200)
        plt.close()
        
        
        
    
    return

def mean_cycle_time_func():
    
    # WT_division_time = []
    # CA_division_time = []
    
    WT_dt = []
    CA_dt = []
    
    for i in range(len(divTimesList)):
        

        dummy_list = list(np.diff(divTimesList[i]))
            
        if cellType[i] == WT_CELL_TYPE:
            WT_dt = WT_dt.copy() + dummy_list.copy()
        elif cellType[i] == CA_CELL_TYPE:
            CA_dt = CA_dt.copy() + dummy_list.copy()
        
        dummy_list.clear()
    
    np.savetxt("pp_data/WT_div_dt.txt", WT_dt, delimiter=',')
    np.savetxt("pp_data/CA_div_dt.txt", CA_dt, delimiter=',')
    
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
    
    
    # Position scatter points and whiskers just above the highest bar
    wt_max_freq = max(np.histogram(WT, bins=15)[0])
    ca_max_freq = max(np.histogram(CA, bins=15)[0])
    scatter_y_pos = max(max_CA, max_WT)*1.1
    scatter_y_pos_c = scatter_y_pos *1.1
    
    # Scatter plot for the means with whiskers on top
    plt.scatter(wt_mean, scatter_y_pos, color='purple', s=100, marker='o', label=f'WT Mean: {wt_mean:.2f}')
    plt.scatter(ca_mean, scatter_y_pos_c, color='green', s=100, marker='o', label=f'CA Mean: {ca_mean:.2f}')
    plt.errorbar(wt_mean, scatter_y_pos, xerr=wt_std, fmt='o', color='purple', capsize=5, label=f'WT ±1 STD: {wt_std:.2f}')
    plt.errorbar(ca_mean, scatter_y_pos_c, xerr=ca_std, fmt='o', color='green', capsize=5, label=f'CA ±1 STD: {ca_std:.2f}')
    
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
    
    plt.savefig("divTimesHist.PNG", dpi=150)
    
    
    return

try:
    directory = "frames"
    os.makedirs(directory, exist_ok=True)
    
    directory = "fit_land"
    os.makedirs(directory, exist_ok=True)
    
    directory = "pp_data"
    os.makedirs(directory, exist_ok=True)
    
    directory = "extra_plot"
    os.makedirs(directory, exist_ok=True)
    
except:
    pass


##### reading params #################################
filename = 'params.csv'
variables = read_custom_csv(filename)
filename = 'pp_params.csv'
pp_variables = read_custom_csv_pp(filename)
frame_plot_switch = pp_variables['frame_plot_switch']

if pp_variables['ent_ext_switch'] and variables['dt_sample']>0.051:
    print('################################')
    print('Warning! dt_sample too large for ent, ext, distributions!')
    print('################################')
    #sys.exit()
##### reading params #################################


####### full temporal hists #############
n_bins_fit = 40
fit_hist_bins = np.linspace(variables['Fit_Th_Apop'], 2.0 * variables['typeFit0'][1], n_bins_fit+1)
fit_hist_bins_focused = np.linspace(variables['typeFit0'][0]-10*variables['typeSigmaFit'][0], variables['typeFit0'][1]+10*variables['typeSigmaFit'][1], n_bins_fit+1)

n_bins_phi = 50
phi_hist_bins = np.linspace(0, 2*np.pi , n_bins_phi+1)

np.savetxt("pp_data"+"/"+"fit_hist_bins.txt", fit_hist_bins, fmt='%1.4f')
np.savetxt("pp_data"+"/"+"fit_hist_bins_focused.txt", fit_hist_bins_focused, fmt='%1.4f')
np.savetxt("pp_data"+"/"+"phi_hist_bins.txt", phi_hist_bins, fmt='%1.4f')
####### full temporal hists #############

####### flux hists #############
Dt_dist = 20
time_window_list = []
time_window_list.append([0, 0+Dt_dist])
time_window_list.append([0.25*variables['maxTime']-Dt_dist/2., 0.25*variables['maxTime']+Dt_dist/2.])
time_window_list.append([0.50*variables['maxTime']-Dt_dist/2., 0.50*variables['maxTime']+Dt_dist/2.])
time_window_list.append([0.75*variables['maxTime']-Dt_dist/2., 0.75*variables['maxTime']+Dt_dist/2.])
time_window_list.append([variables['maxTime']-Dt_dist, variables['maxTime']])

# phiSections = np.array([0.02, 0.98]) * 2 * np.pi
phiSections = np.array([0.02, variables['G1Border']+0.01]) * 2 * np.pi

# enter_fit_data = len(time_window_list) * [[]]
# exit_fit_data = len(time_window_list) * [[]]

enter_fit_data = [[] for _ in range(len(time_window_list))]
exit_fit_data = [[] for _ in range(len(time_window_list))]

####### flux hists #############



######################### Load Params ############################
cellX = np.loadtxt('init/X_init.txt', delimiter=',')
cellY = np.loadtxt('init/Y_init.txt', delimiter=',')
cellPhi = np.loadtxt('init/Phi_init.txt', delimiter=',')
cellR  = np.loadtxt('init/R_init.txt', delimiter=',')
cellType = np.loadtxt('init/Type_init.txt', delimiter=',', dtype=int)
cellState = np.loadtxt('init/State_init.txt', delimiter=',', dtype=int)
cellFitness = np.loadtxt('init/Fit_init.txt', delimiter=',', dtype=float)
NCells = len(cellX)

A_min_types = np.zeros(variables['NTypes'])
A_max_types = np.zeros(variables['NTypes'])

alive_stat = np.zeros(1+variables['samplesPerWrite'], dtype=int)
C_alive_stat = np.zeros(1+variables['samplesPerWrite'], dtype=int)
C_dead_stat = np.zeros(1+variables['samplesPerWrite'], dtype=int)
WT_alive_stat = np.zeros(1+variables['samplesPerWrite'], dtype=int)
WT_cyc_stat = np.zeros(1+variables['samplesPerWrite'], dtype=int)
WT_g1_cyc_stat = np.zeros(1+variables['samplesPerWrite'], dtype=int)
WT_g1_arr_stat = np.zeros(1+variables['samplesPerWrite'], dtype=int)
WT_g1_tot_stat = np.zeros(1+variables['samplesPerWrite'], dtype=int)
WT_sg2m_stat =   np.zeros(1+variables['samplesPerWrite'], dtype=int)
WT_g0_stat = np.zeros(1+variables['samplesPerWrite'], dtype=int)
WT_diff_stat = np.zeros(1+variables['samplesPerWrite'], dtype=int)
WT_dead_stat = np.zeros(1+variables['samplesPerWrite'], dtype=int)

WT_fit_hist_data = np.zeros((n_bins_fit, 1+variables['samplesPerWrite']), dtype=float)
CA_fit_hist_data = np.zeros((n_bins_fit, 1+variables['samplesPerWrite']), dtype=float)
WT_phi_hist_data = np.zeros((n_bins_phi, 1+variables['samplesPerWrite']), dtype=float)
CA_phi_hist_data = np.zeros((n_bins_phi, 1+variables['samplesPerWrite']), dtype=float)


for typeC in range(variables['NTypes']):
    
    r0   = variables['typeR0'][typeC]
    r2PI = variables['typeR2PI'][typeC]
    
    A_min_types[typeC] = np.pi*r0**2
    A_max_types[typeC] = np.pi*r2PI**2
    
    
# cellR = 0.0 * cellPhi
# for cellC in range(NCells):
#     cellType_val = cellType[cellC]
#     # r0   = variables['typeR0'][cellType_val]
#     # r2PI = variables['typeR2PI'][cellType_val]
#     # A_min = np.pi*r0**2
#     # A_max = np.pi*r2PI**2
#     area_val = A_min_types[cellType_val] + (A_max_types[cellType_val] - A_min_types[cellType_val]) * cellPhi[cellC] / (2 * np.pi)
#     cellR[cellC] = (area_val / np.pi)**0.5
    
    
    
    # WT_alive_stat = np.zeros(1+variables['samplesPerWrite'])
    # WT_cyc_stat = np.zeros(1+variables['samplesPerWrite'])
    # WT_g1_cyc_stat = np.zeros(1+variables['samplesPerWrite'])
    # WT_g1_arr_stat = np.zeros(1+variables['samplesPerWrite'])
    # WT_g1_tot_stat = np.zeros(1+variables['samplesPerWrite'])
    # WT_g0_stat = np.zeros(1+variables['samplesPerWrite'])
    # WT_diff_stat = np.zeros(1+variables['samplesPerWrite'])
    # WT_apop_stat = np.zeros(1+variables['samplesPerWrite'])

C_alive_stat[0] =   len(cellType[(cellType==CA_CELL_TYPE) & (cellState==CYCLING_STATE)])
C_dead_stat[0] =    len(cellType[(cellType==CA_CELL_TYPE) & (cellState<=APOP_STATE+0.01)])
WT_dead_stat[0] =   len(cellType[(cellType==WT_CELL_TYPE) & (cellState<=APOP_STATE+0.01)])
WT_alive_stat[0] =  len(cellType[(cellType==WT_CELL_TYPE) & (cellState > APOP_STATE)])
alive_stat[0] =     C_alive_stat[0] + WT_alive_stat[0]
WT_diff_stat[0] =   len(cellState[cellState==DIFF_STATE])
WT_g0_stat[0] =     len(cellState[cellState==G0_STATE])
WT_g1_arr_stat[0] = len(cellState[cellState==G1_ARR_STATE])
WT_cyc_stat[0] =    len(cellType[(cellType==WT_CELL_TYPE) & (cellState==CYCLING_STATE)])
WT_g1_cyc_stat[0] = len(cellState[(cellType==WT_CELL_TYPE) & (cellState==CYCLING_STATE) & (cellPhi<=2.0*np.pi*variables['G1Border'])])
WT_g1_tot_stat[0] = WT_g1_cyc_stat[0] + WT_g1_arr_stat[0]
WT_sg2m_stat[0] = len(cellState[(cellType==WT_CELL_TYPE) & (cellState==CYCLING_STATE) & (cellPhi>2.0*np.pi*variables['G1Border'])])

WT_fit_hist_data[:,0] = np.histogram(cellFitness[:,0][(cellType==WT_CELL_TYPE) & (cellState > APOP_STATE)], bins = fit_hist_bins, density=True)[0]
CA_fit_hist_data[:,0] = np.histogram(cellFitness[:,0][(cellType==CA_CELL_TYPE) & (cellState > APOP_STATE)], bins = fit_hist_bins, density=True)[0]
WT_phi_hist_data[:,0] = np.histogram(cellPhi[(cellType==WT_CELL_TYPE) & (cellState > APOP_STATE)], bins = phi_hist_bins, density=True)[0]
CA_phi_hist_data[:,0] = np.histogram(cellPhi[(cellType==CA_CELL_TYPE) & (cellState > APOP_STATE)], bins = phi_hist_bins, density=True)[0]





# Cancer = np.array([np.sum(cellType)])
# WT = np.array([NCells - np.sum(cellType)])



time = np.array([0])

snapshotInd = 0
if frame_plot_switch:
    plotter(0.0, snapshotInd)
    landscape_plotter(0.0, snapshotInd)
snapshotInd += 1


args = ['all']
subprocess.run(['python3', 'dataUnzipper.py'] + args)

dt = variables['dt']

bunchInd = 1
t = dt


divTimesList = []




######## Simulation loop ###########
while(1):
    
    
    try:
        ## for divTimes files
        ## The bunch index here is not like the other data. Be careful! It is only a file index.
        with open("data/divisionTimes_"+str(bunchInd)+".txt", 'r') as divTimesFile:
            lineC = -1
            for line in divTimesFile:
                lineC += 1
                
                stripped_line = line.strip()
                
                if lineC >= len(divTimesList):
                    if (line.strip()): # not empty
                        numbers = [float(num) for num in stripped_line.split(',')]
                        divTimesList.append(numbers)
                    else: # empty line
                        divTimesList.append([])
                else:
                    if not (line.strip()): # the line is empty
                        pass
                    else:
                        numbers = [float(num) for num in stripped_line.split(',')]
                        divTimesList[lineC] = divTimesList[lineC].copy() + numbers.copy()
        divTimesFile.close()
        ## for divTimes files
        
        t_bunch = np.loadtxt('data/t_'+str(bunchInd)+'.txt', delimiter=',')
        X_bunch = np.loadtxt('data/X_'+str(bunchInd)+'.txt', delimiter=',')
        Y_bunch = np.loadtxt('data/Y_'+str(bunchInd)+'.txt', delimiter=',')
        type_bunch = np.loadtxt('data/Type_'+str(bunchInd)+'.txt', delimiter=',', dtype=int)
        state_bunch = np.loadtxt('data/State_'+str(bunchInd)+'.txt', delimiter=',', dtype=int)
        Phi_bunch = np.loadtxt('data/Phi_'+str(bunchInd)+'.txt', delimiter=',')
        R_bunch = np.loadtxt('data/R_'+str(bunchInd)+'.txt', delimiter=',')
        fitness_bunch = np.loadtxt('data/Fit_'+str(bunchInd)+'.txt', delimiter=',')
        
        if bunchInd > 1:
            zeros_to_append = np.zeros(variables['samplesPerWrite'], dtype=int)
            fit_hist_zeros_to_append = np.zeros((n_bins_fit, variables['samplesPerWrite']))
            phi_hist_zeros_to_append = np.zeros((n_bins_phi, variables['samplesPerWrite']))
            
            alive_stat = np.append(alive_stat, zeros_to_append.copy())
            C_alive_stat =     np.append(C_alive_stat, zeros_to_append.copy())
            C_dead_stat =     np.append(C_dead_stat, zeros_to_append.copy())
            WT_alive_stat = np.append(WT_alive_stat, zeros_to_append.copy())
            WT_cyc_stat = np.append(WT_cyc_stat, zeros_to_append.copy())
            WT_g1_cyc_stat = np.append(WT_g1_cyc_stat, zeros_to_append.copy())
            WT_g1_arr_stat = np.append(WT_g1_arr_stat, zeros_to_append.copy())
            WT_g1_tot_stat = np.append(WT_g1_tot_stat, zeros_to_append.copy())
            WT_g0_stat = np.append(WT_g0_stat, zeros_to_append.copy())
            WT_diff_stat = np.append(WT_diff_stat, zeros_to_append.copy())
            WT_dead_stat = np.append(WT_dead_stat, zeros_to_append.copy())
            WT_sg2m_stat = np.append(WT_sg2m_stat, zeros_to_append.copy())
            
            WT_fit_hist_data = np.append(WT_fit_hist_data, fit_hist_zeros_to_append.copy(), axis=1)
            CA_fit_hist_data = np.append(CA_fit_hist_data, fit_hist_zeros_to_append.copy(), axis=1)
            WT_phi_hist_data = np.append(WT_phi_hist_data, phi_hist_zeros_to_append.copy(), axis=1)
            CA_phi_hist_data = np.append(CA_phi_hist_data, phi_hist_zeros_to_append.copy(), axis=1)
            
        # try:
        #     cellX = np.loadtxt('data/X_'+str(ind)+'.txt', delimiter=',')
        #     cellY = np.loadtxt('data/Y_'+str(ind)+'.txt', delimiter=',')
        #     cellR = np.loadtxt('data/R_'+str(ind)+'.txt', delimiter=',')
        #     cellType = np.loadtxt('data/Type_'+str(ind)+'.txt', delimiter=',', dtype=int)
        #     cellFitness = np.loadtxt('data/Fitness_'+str(ind)+'.txt', delimiter=',', dtype=float)
        #     NCells = len(cellX)
        
        for sampleC in range(len(t_bunch)):
            t = t_bunch[sampleC]
            
            cellType_old = cellType.copy()
            
            cellType = type_bunch[type_bunch[:, sampleC] > -0.5, sampleC]
            NCells = len(cellType)
            cellX = X_bunch[:NCells, sampleC]
            cellY = Y_bunch[:NCells, sampleC]
            
            
            for t_window_C in range(len(time_window_list)):
                t_min_window = time_window_list[t_window_C][0]
                t_max_window = time_window_list[t_window_C][1]
                if t>t_min_window and t<t_max_window:
                    
                    NCells_old = len(cellPhi)
                    cellFitness_real_old = cellFitness[:,0]
                    condition_enter = \
                    (Phi_bunch[:NCells_old, sampleC]>phiSections[0]) & \
                    (cellPhi < phiSections[0]) & (cellState > APOP_STATE) & \
                    (cellType_old == WT_CELL_TYPE)
                    
                    enter_fit_data[t_window_C]= enter_fit_data[t_window_C]+list(cellFitness_real_old[condition_enter])
                    
                    condition_exit = \
                    (Phi_bunch[:NCells_old, sampleC]>phiSections[1]) & \
                    (cellPhi < phiSections[1]) & (cellState > APOP_STATE) & \
                    (cellType_old == WT_CELL_TYPE)
                    
                    exit_fit_data[t_window_C]= exit_fit_data[t_window_C]+list(cellFitness_real_old[condition_exit])
                    
                else:
                    pass
                
            cellState = state_bunch[:NCells, sampleC]
            cellFitness = fitness_bunch[:NCells, 2*sampleC:2*sampleC+2]
            cellPhi = Phi_bunch[:NCells, sampleC]
            
            # cellR = 0.0 * cellPhi
            # for cellC in range(NCells):
            #     cellType_val = cellType[cellC]
            #     # r0   = variables['typeR0'][cellType_val]
            #     # r2PI = variables['typeR2PI'][cellType_val]
            #     # A_min = np.pi*r0**2
            #     # A_max = np.pi*r2PI**2
            #     # area_val = A_min + (A_max - A_min) * cellPhi[cellC] / (2 * np.pi)
            #     area_val = A_min_types[cellType_val] + (A_max_types[cellType_val] - A_min_types[cellType_val]) * cellPhi[cellC] / (2 * np.pi)
            #     cellR[cellC] = (area_val / np.pi)**0.5
            
            cellR = R_bunch[:NCells, sampleC]
            
            
            # Cancer = np.append(Cancer, np.sum(cellType))
            # WT = np.append(WT, NCells - np.sum(cellType))
            
            C_alive_stat[snapshotInd] =   len(cellType[(cellType==CA_CELL_TYPE) & (cellState > APOP_STATE)])
            C_dead_stat[snapshotInd] =    len(cellType[(cellType==CA_CELL_TYPE) & (cellState<=APOP_STATE+0.01)])
            WT_dead_stat[snapshotInd] =   len(cellType[(cellType==WT_CELL_TYPE) & (cellState<=APOP_STATE+0.01)])
            WT_alive_stat[snapshotInd] =  len(cellType[(cellType==WT_CELL_TYPE) & (cellState > APOP_STATE)])
            alive_stat[snapshotInd] =     C_alive_stat[snapshotInd] + WT_alive_stat[snapshotInd]
            WT_diff_stat[snapshotInd] =   len(cellState[cellState==DIFF_STATE])
            WT_g0_stat[snapshotInd] =     len(cellState[cellState==G0_STATE])
            WT_g1_arr_stat[snapshotInd] = len(cellState[cellState==G1_ARR_STATE])
            WT_cyc_stat[snapshotInd] =    len(cellType[(cellType==WT_CELL_TYPE) & (cellState==CYCLING_STATE)])
            WT_g1_cyc_stat[snapshotInd] = len(cellState[(cellType==WT_CELL_TYPE) & (cellState==CYCLING_STATE) & (cellPhi<=2.0*np.pi*variables['G1Border'])])
            WT_g1_tot_stat[snapshotInd] = WT_g1_cyc_stat[snapshotInd] + WT_g1_arr_stat[snapshotInd]
            WT_sg2m_stat[snapshotInd] =   len(cellState[(cellType==WT_CELL_TYPE) & (cellState==CYCLING_STATE) & (cellPhi>2.0*np.pi*variables['G1Border'])])
            
                
            WT_fit_hist_data[:,snapshotInd] = np.histogram(cellFitness[:,0][(cellType==WT_CELL_TYPE) & (cellState > APOP_STATE)], bins = fit_hist_bins, density=True)[0]
            CA_fit_hist_data[:,snapshotInd] = np.histogram(cellFitness[:,0][(cellType==CA_CELL_TYPE) & (cellState > APOP_STATE)], bins = fit_hist_bins, density=True)[0]
            WT_phi_hist_data[:,snapshotInd] = np.histogram(cellPhi[(cellType==WT_CELL_TYPE) & (cellState > APOP_STATE)], bins = phi_hist_bins, density=True)[0]
            CA_phi_hist_data[:,snapshotInd] = np.histogram(cellPhi[(cellType==CA_CELL_TYPE) & (cellState > APOP_STATE)], bins = phi_hist_bins, density=True)[0]
            
            
            
            
            
            time = np.append(time, t)

            if frame_plot_switch:
                plotter(t, snapshotInd)
                landscape_plotter(t, snapshotInd)
                
            snapshotInd += 1
            # t += dt
            print(t)
        
        bunchInd += 1
        
    except:
        break

np.savetxt("pp_data"+"/"+"time.txt", time, fmt='%.4f')
np.savetxt("pp_data"+"/"+"alive_stat.txt", alive_stat, fmt='%d')
np.savetxt("pp_data"+"/"+"C_alive_stat.txt", C_alive_stat, fmt='%d')
np.savetxt("pp_data"+"/"+"C_dead_stat.txt", C_dead_stat, fmt='%d')
np.savetxt("pp_data"+"/"+"WT_alive_stat.txt", WT_alive_stat, fmt='%d')
np.savetxt("pp_data"+"/"+"WT_cyc_stat.txt", WT_cyc_stat, fmt='%d')
np.savetxt("pp_data"+"/"+"WT_g1_cyc_stat.txt", WT_g1_cyc_stat, fmt='%d')
np.savetxt("pp_data"+"/"+"WT_g1_arr_stat.txt", WT_g1_arr_stat, fmt='%d')
np.savetxt("pp_data"+"/"+"WT_g1_tot_stat.txt", WT_g1_tot_stat, fmt='%d')
np.savetxt("pp_data"+"/"+"WT_g0_stat.txt", WT_g0_stat, fmt='%d')
np.savetxt("pp_data"+"/"+"WT_diff_stat.txt", WT_diff_stat, fmt='%d')
np.savetxt("pp_data"+"/"+"WT_dead_stat.txt", WT_dead_stat, fmt='%d')
np.savetxt("pp_data"+"/"+"WT_sg2m_stat.txt", WT_sg2m_stat, fmt='%d')

np.savetxt("pp_data"+"/"+"WT_fit_hist_data.txt", WT_fit_hist_data, fmt='%1.4f', delimiter=',')
np.savetxt("pp_data"+"/"+"CA_fit_hist_data.txt", CA_fit_hist_data, fmt='%1.4f', delimiter=',')
np.savetxt("pp_data"+"/"+"WT_phi_hist_data.txt", WT_phi_hist_data, fmt='%1.4f', delimiter=',')
np.savetxt("pp_data"+"/"+"CA_phi_hist_data.txt", CA_phi_hist_data, fmt='%1.4f', delimiter=',')

if pp_variables['ent_ext_switch']:
    ent_ext_fitness_dist_saver()

stats_plotter("statistics")

hist_plotter()

mean_cycle_time_func()
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


CYCLING_STATE =   (1)
G1_ARR_STATE =    (-1)
G0_STATE =        (-2)
DIFF_STATE =      (-3)
APOP_STATE =      (-4)
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
    
    typeFit0 = variables.get('typeFit0', None)
    Fit_Th_G1_arr = variables.get('Fit_Th_G1_arr', None)
    Fit_Th_G0 = variables.get('Fit_Th_G0', None)
    Fit_Th_Diff = variables.get('Fit_Th_Diff', None)
    Fit_Th_Apop = variables.get('Fit_Th_Apop', None)
    
    maxTime = variables.get('maxTime', None)
    dt = variables.get('dt', None)
    dt_sample = variables.get('dt_sample', None)
    samplesPerWrite = variables.get('samplesPerWrite', None)
    printingTimeInterval = variables.get('printingTimeInterval', None)
    
    R_cut_coef_game = variables.get('R_cut_coef_game', None)
    GameNoiseSigma = variables.get('GameNoiseSigma', None)
    tau = variables.get('tau', None)
    typeTypePayOff_mat_real_C = variables.get('typeTypePayOff_mat_real_C', None)
    typeTypePayOff_mat_real_F1 = variables.get('typeTypePayOff_mat_real_F1', None)
    typeTypePayOff_mat_real_F2 = variables.get('typeTypePayOff_mat_real_F2', None)
    typeTypePayOff_mat_imag_C = variables.get('typeTypePayOff_mat_imag_C', None)
    typeTypePayOff_mat_imag_F1 = variables.get('typeTypePayOff_mat_imag_F1', None)
    typeTypePayOff_mat_imag_F2 = variables.get('typeTypePayOff_mat_imag_F2', None)
    
    # typeOmega0 = variables.get('typeOmega0', None)
    # typeOmegaLim = variables.get('typeOmegaLim', None)
    
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
        
        'typeFit0' : typeFit0,
        'Fit_Th_G1_arr' : Fit_Th_G1_arr,
        'Fit_Th_G0' : Fit_Th_G0,
        'Fit_Th_Diff' : Fit_Th_Diff,
        'Fit_Th_Apop' : Fit_Th_Apop,
        
        'maxTime': maxTime,
        'dt': dt,
        'dt_sample': dt_sample,
        'samplesPerWrite': samplesPerWrite,
        'printingTimeInterval': printingTimeInterval,
        
        'R_cut_coef_game': R_cut_coef_game,
        
        'R_cut_coef_game' : R_cut_coef_game,
        'GameNoiseSigma' : GameNoiseSigma,
        'tau' :tau,
        'typeTypePayOff_mat_real_C' :typeTypePayOff_mat_real_C,
        'typeTypePayOff_mat_real_F1' :typeTypePayOff_mat_real_F1,
        'typeTypePayOff_mat_real_F2' :typeTypePayOff_mat_real_F2,
        'typeTypePayOff_mat_imag_C' :typeTypePayOff_mat_imag_C,
        'typeTypePayOff_mat_imag_F1' :typeTypePayOff_mat_imag_F1,
        'typeTypePayOff_mat_imag_F2' :typeTypePayOff_mat_imag_F2,
        
        'initConfig': initConfig
    }


def plotter(t, snapshotInd):
    # size_vec = np.zeros(N_sph_tot)
    # scale = 4

    # norm = mcolors.Normalize(vmin=cell_sph.min(), vmax=cell_sph.max())
    # cmap = cm.viridis
    
    # fig, ax = plt.subplots()
    
    fig, (ax1, ax2) = plt.subplots(1, 2)
    
    WT_indices = (cellType==0)
    C_indices = (cellType==1)
    
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
        
        if cellState[i]==APOP_STATE:
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
    
    ax2.plot(time, WT/WT[0], label='WT(t)/WT(0)', color='violet')
    ax2.plot(time, Cancer/Cancer[0], label='C(t)/C(0)', color='g')
    ax2.plot(time, Cancer/(Cancer+WT), label='C(t)/tot(t)', color='g', linestyle='dashed')
    ax2.plot(time, WT/(Cancer+WT), label='WT(t)/tot(t)', color='violet', linestyle='dashed')
    ax2.set_yscale("log")
    
    ax2.legend()
    ax2.grid()
    ax2.set_xlabel('time')
    
    
    # ax1.set_ylabel('Y-axis')
    
    plt.savefig(file_name, dpi=150)
    # plt.show()
    plt.close()
    
    return 0


try:
    directory = "frames"
    os.makedirs(directory, exist_ok=True)
except:
    pass

filename = 'params.csv'
variables = read_custom_csv(filename)

######################### Load Params ############################
cellX = np.loadtxt('init/X_init.txt', delimiter=',')
cellY = np.loadtxt('init/Y_init.txt', delimiter=',')
cellPhi = np.loadtxt('init/Phi_init.txt', delimiter=',')
cellType = np.loadtxt('init/Type_init.txt', delimiter=',', dtype=int)
cellState = np.loadtxt('init/State_init.txt', delimiter=',', dtype=int)
cellFitness = np.loadtxt('init/Fit_init.txt', delimiter=',', dtype=float)
NCells = len(cellX)

A_min_types = np.zeros(variables['NTypes'])
A_max_types = np.zeros(variables['NTypes'])

for typeC in range(variables['NTypes']):
    
    r0   = variables['typeR0'][typeC]
    r2PI = variables['typeR2PI'][typeC]
    
    A_min_types[typeC] = np.pi*r0**2
    A_max_types[typeC] = np.pi*r2PI**2
    
cellR = 0.0 * cellPhi
for cellC in range(NCells):
    cellType_val = cellType[cellC]
    # r0   = variables['typeR0'][cellType_val]
    # r2PI = variables['typeR2PI'][cellType_val]
    
    # A_min = np.pi*r0**2
    # A_max = np.pi*r2PI**2
    
    area_val = A_min_types[cellType_val] + (A_max_types[cellType_val] - A_min_types[cellType_val]) * cellPhi[cellC] / (2 * np.pi)
    
    cellR[cellC] = (area_val / np.pi)**0.5
    
    

Cancer = np.array([np.sum(cellType)])
WT = np.array([NCells - np.sum(cellType)])
time = np.array([0])

snapshotInd = 0
plotter(0.0, snapshotInd)
snapshotInd += 1

args = ['all']
subprocess.run(['python3', 'dataUnzipper.py'] + args)

dt = variables['dt']

bunchInd = 1
t = dt
######## Simulation loop ###########
while(1):
    
    try:
        t_bunch = np.loadtxt('data/t_'+str(bunchInd)+'.txt', delimiter=',')
        X_bunch = np.loadtxt('data/X_'+str(bunchInd)+'.txt', delimiter=',')
        Y_bunch = np.loadtxt('data/Y_'+str(bunchInd)+'.txt', delimiter=',')
        type_bunch = np.loadtxt('data/Type_'+str(bunchInd)+'.txt', delimiter=',', dtype=int)
        state_bunch = np.loadtxt('data/State_'+str(bunchInd)+'.txt', delimiter=',', dtype=int)
        Phi_bunch = np.loadtxt('data/Phi_'+str(bunchInd)+'.txt', delimiter=',')
        fitness_bunch = np.loadtxt('data/Fit_'+str(bunchInd)+'.txt', delimiter=',')
        
        # try:
        #     cellX = np.loadtxt('data/X_'+str(ind)+'.txt', delimiter=',')
        #     cellY = np.loadtxt('data/Y_'+str(ind)+'.txt', delimiter=',')
        #     cellR = np.loadtxt('data/R_'+str(ind)+'.txt', delimiter=',')
        #     cellType = np.loadtxt('data/Type_'+str(ind)+'.txt', delimiter=',', dtype=int)
        #     cellFitness = np.loadtxt('data/Fitness_'+str(ind)+'.txt', delimiter=',', dtype=float)
        #     NCells = len(cellX)
        
        for sampleC in range(len(t_bunch)):
            t = t_bunch[sampleC]
            
            cellType = type_bunch[type_bunch[:, sampleC] > -0.5, sampleC]
            NCells = len(cellType)
            cellX = X_bunch[:NCells, sampleC]
            cellY = Y_bunch[:NCells, sampleC]
            cellPhi = Phi_bunch[:NCells, sampleC]
            cellState = state_bunch[:NCells, sampleC]
            cellFitness = fitness_bunch[:NCells, 2*sampleC:2*sampleC+2]
            
            cellR = 0.0 * cellPhi
            for cellC in range(NCells):
                cellType_val = cellType[cellC]
                # r0   = variables['typeR0'][cellType_val]
                # r2PI = variables['typeR2PI'][cellType_val]
                
                # A_min = np.pi*r0**2
                # A_max = np.pi*r2PI**2
                
                # area_val = A_min + (A_max - A_min) * cellPhi[cellC] / (2 * np.pi)
                area_val = A_min_types[cellType_val] + (A_max_types[cellType_val] - A_min_types[cellType_val]) * cellPhi[cellC] / (2 * np.pi)
                
                cellR[cellC] = (area_val / np.pi)**0.5
            
            
            Cancer = np.append(Cancer, np.sum(cellType))
            # WT = np.append(WT, NCells - np.sum(cellType))
            WT = np.append(WT, NCells - np.sum(cellType)-len(cellState[cellState==APOP_STATE]))
            
            time = np.append(time, t)


            plotter(t, snapshotInd)
            snapshotInd += 1
            # t += dt
            print(t)
        
        bunchInd += 1
        
    except:
        break




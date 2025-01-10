#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 16:50:26 2024

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


initStepsFolderName = "initialize_steps"


##### reading params #################################
filename = 'params.csv'
variables = read_custom_csv(filename)
##### reading params #################################


stepIndex = -1
while(1):
    stepIndex += 1
    try:
        cellType  = np.loadtxt(initStepsFolderName+'/Type_'+str(stepIndex)+'.txt', delimiter=',', dtype=int)
        cellState = np.loadtxt(initStepsFolderName+'/State_'+str(stepIndex)+'.txt', delimiter=',', dtype=int)
        cellPhi   = np.loadtxt(initStepsFolderName+'/Phi_'+str(stepIndex)+'.txt', delimiter=',', dtype=float)
        cellR     = np.loadtxt(initStepsFolderName+'/R_'+str(stepIndex)+'.txt', delimiter=',', dtype=float)
        cellFitness   = np.loadtxt(initStepsFolderName+'/Fit_'+str(stepIndex)+'.txt', delimiter=',', dtype=float)
    except FileNotFoundError:
        break
    
init_numbers = np.loadtxt('Init_numbers.csv', delimiter=',', dtype=int)

WT_alive_cells = []
CA_alive_cells = []

for cellC in range(len(cellType)):
    
    if cellState[cellC] > APOP_STATE:
        
        if cellType[cellC]==WT_CELL_TYPE:
            WT_alive_cells.append(cellC)
        elif cellType[cellC]==CA_CELL_TYPE:
            CA_alive_cells.append(cellC)
        

sampled_WT = np.random.choice(WT_alive_cells, init_numbers[WT_CELL_TYPE])
sampled_CA = np.random.choice(CA_alive_cells, init_numbers[CA_CELL_TYPE])

Fitness_Real = cellFitness[:,0]



if len(sampled_WT)>0:
    sampled_phi_WT = cellPhi[sampled_WT]
    sampled_R_WT = cellR[sampled_WT]
    sampled_state_WT = cellState[sampled_WT]
    sampled_fit_real_WT = Fitness_Real[sampled_WT]
else:
    sampled_phi_WT = np.array([])
    sampled_R_WT = np.array([])
    sampled_state_WT = np.array([])
    sampled_fit_real_WT = np.array([])
    
if len(sampled_CA)>0:
    sampled_phi_CA = cellPhi[sampled_CA]
    sampled_R_CA = cellR[sampled_CA]
    sampled_state_CA = cellState[sampled_CA]
    sampled_fit_real_CA = Fitness_Real[sampled_CA]
else:
    sampled_phi_CA = np.array([])
    sampled_R_CA = np.array([])
    sampled_state_CA = np.array([])
    sampled_fit_real_CA = np.array([])

## PHI
sampled_phi_tot = np.concatenate((sampled_phi_WT, sampled_phi_CA))
np.savetxt("init"+"/"+"Phi_init.txt", sampled_phi_tot, fmt='%1.6f', delimiter=',')
## phi

## R
sampled_R_tot = np.concatenate((sampled_R_WT, sampled_R_CA))
np.savetxt("init"+"/"+"R_init.txt", sampled_R_tot, fmt='%1.6f', delimiter=',')
## R

## State
sampled_state_tot = np.concatenate((sampled_state_WT, sampled_state_CA))
np.savetxt("init"+"/"+"State_init.txt", sampled_state_tot, fmt='%d', delimiter=',')
## State


## Fitness
sampled_fit_real_tot = np.concatenate((sampled_fit_real_WT, sampled_fit_real_CA))
sampled_fit_tot = np.zeros((len(sampled_fit_real_tot),2))
sampled_fit_tot[:,0] = sampled_fit_real_tot
np.savetxt("init"+"/"+"Fit_init.txt", sampled_fit_tot, fmt='%1.6f', delimiter=',')
## Fitness









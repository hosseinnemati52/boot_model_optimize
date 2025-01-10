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


#################### FUNCTIONS ####################
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


def PhiFR_init():
    
    cellType = []
    
    for typeC in range(variables['NTypes']):
        if Init_numbers[typeC] > 0:
            cellType = cellType + Init_numbers[typeC]*[typeC]
    
    cellType = np.array(cellType, dtype=int)
    
    cellR       = 0.0 * np.ones(NCells)
    cellArea    = 0.0 * np.ones(NCells)
    cellFitness = 0.0 * np.ones((NCells,2))
    
    cellPhi = np.random.uniform(low=0.0, high=2*np.pi, size=NCells)
    
    for cellC in range(NCells):
        
        A_min = np.pi * (variables['typeR0'][cellType[cellC]]) **2
        A_max = np.pi * (variables['typeR2PI'][cellType[cellC]]) **2
        A_val = A_min + (A_max-A_min) * cellPhi[cellC]/(2*np.pi)
        
        cellArea[cellC] = A_val
        cellR[cellC] = (A_val/np.pi) ** 0.5
        
        cellFitness[cellC,0] = variables['typeFit0'][cellType[cellC]]
        cellFitness[cellC,1] = 0.0
    
    cellState   = CYCLING_STATE * np.ones(NCells, dtype=int)
    
    return cellType, cellPhi, cellR, cellArea, cellState, cellFitness

def XYVxVy_init():
    
    cellX  = 0.0 * np.ones(NCells)
    cellY  = 0.0 * np.ones(NCells)
    cellVx = 0.0 * np.ones(NCells)
    cellVy = 0.0 * np.ones(NCells)
    
    typeArea = 0.0 * np.ones(variables['NTypes'])
    
    for cellC in range(NCells):
        typeArea[cellType[cellC]] += (cellArea[cellC])
    
    
    # hex_packing_2d = 0.92
    # hex_packing_2d = 0.9069
    hex_packing_2d = 1.05
    # hex_packing_2d = 0.65
    plus_factor = 0.01
    allowed_overlap = 0.4
    
    A_total = np.sum(cellArea)
    
    A_total_extra = (A_total/hex_packing_2d)/(1+plus_factor)
    
    typeArea_extra = (typeArea/np.sum(typeArea)) * A_total_extra
    
    n_Init_thresh = 5
    # min_numbers = np.min(Init_numbers)
    
    N_tries = 10 * NCells
    while_cond = 1
    
    while(1):
        A_total_extra *= (1+plus_factor)
        print(A_total_extra)
        R_total_extra = (A_total_extra/np.pi)**0.5
        
        if Init_numbers[0] == 0:
            h = -R_total_extra
        elif Init_numbers[1] == 0:
            h =  R_total_extra
        else:
            alpha = np.linspace(0, np.pi, 1000)
            A_0_candidate = (np.pi-alpha+np.sin(alpha)*np.cos(alpha)) * R_total_extra**2
            A_1_candidate = np.pi * R_total_extra**2 - A_0_candidate
            
            err = (A_1_candidate-typeArea_extra[1])**2 + \
            (A_0_candidate-typeArea_extra[0])**2
            
            alpha_star = alpha[np.argmin(err)]
            
            h = R_total_extra * np.cos(alpha_star)
            
        switch_bigger_area = 0
        
        placed_cells = dict()
        placed_cells[WT_CELL_TYPE] = []
        placed_cells[CA_CELL_TYPE] = []
        placed_cells[WT_CELL_TYPE].clear()
        placed_cells[CA_CELL_TYPE].clear()
        placed_cells[WT_CELL_TYPE] = []
        placed_cells[CA_CELL_TYPE] = []
        
        for cellC_1 in range(NCells):
            condCell = 0
            tryC = 0
            while (condCell == 0):
                
                if cellC_1 == 0: # the first cell in general
                    x_min = -h*(1-(h/R_total_extra)**2)**0.5
                    x_max = +h*(1-(h/R_total_extra)**2)**0.5
                    if cellType[cellC_1]==WT_CELL_TYPE:
                        y_max = h - (1-allowed_overlap)*cellR[cellC_1]
                        y_min = h - (1-allowed_overlap/2)*cellR[cellC_1]
                    elif cellType[cellC_1]==CA_CELL_TYPE:
                        y_min = h + (1-allowed_overlap)*cellR[cellC_1]
                        y_max = h + (1-allowed_overlap/2)*cellR[cellC_1]
                    
                    cellX[cellC_1]= np.random.uniform(low=x_min, high=x_max, size=1)
                    cellY[cellC_1]= np.random.uniform(low=y_min, high=y_max, size=1)
                    
                else: 
                    
                    # if len(placed_cells[cellType[cellC_1]])==0: # The first cancer cell, maybe
                    #     bead_cell = np.random.randint(0, cellC_1, 1)
                    # else:
                    #     bead_cell = np.random.choice(placed_cells[cellType[cellC_1]])
                    bead_cell = np.random.randint(0, cellC_1, 1)
                    
                    # angle  = np.random.uniform(low=0.0, high=2*np.pi, size=1)
                    # c_to_c = np.random.uniform(low=(1-allowed_overlap)*(cellR[cellC_1]+cellR[bead_cell]), high=(cellR[cellC_1]+cellR[bead_cell]), size=1)
                    
                    # cellX[cellC_1]= cellX[bead_cell] + c_to_c * np.cos(angle)
                    # cellY[cellC_1]= cellY[bead_cell] + c_to_c * np.sin(angle)
                    
                    cellX[cellC_1]= np.random.uniform(low=-R_total_extra, high=R_total_extra, size=1)
                    cellY[cellC_1]= np.random.uniform(low=-R_total_extra, high=R_total_extra, size=1)
                
                condSeparation = \
                ( (cellType[cellC_1]==1) and (cellY[cellC_1]>h) )\
                or \
                ( (cellType[cellC_1]==0) and (cellY[cellC_1]<h) )
                
                if not condSeparation:
                    continue
                
                # if Init_numbers[cellType[cellC_1]] == min_numbers:
                if Init_numbers[cellType[cellC_1]] < n_Init_thresh:
                    condInside = cellY[cellC_1] < R_total_extra
                    condInside = (cellX[cellC_1]**2 + cellY[cellC_1]**2 < R_total_extra**2)
                    # condInside = 1
                else:
                    condInside = (cellX[cellC_1]**2 + cellY[cellC_1]**2 < R_total_extra**2)
                    # condInside = 1
                    
                
                
                if not condInside:
                    continue
                
                
                condOverlap = 0
                for cellC_2 in range(cellC_1):
                    distance = ((cellX[cellC_1]-cellX[cellC_2])**2 \
                              + (cellY[cellC_1]-cellY[cellC_2])**2) ** 0.5
                    if distance < (1-allowed_overlap) * (cellR[cellC_1]+cellR[cellC_2]):
                        condOverlap = 1
                        tryC += 1
                        break
                
                if condOverlap:
                    continue
                
                condCell =  (condInside) and (condSeparation) and (not condOverlap)

                # if tryC>N_tries:
                #     switch_bigger_area = 1
                #     break
            
            placed_cells[cellType[cellC_1]].append(cellC_1)
            
            # if switch_bigger_area:
            #     break
        
        # while_cond = 0
        if (cellC_1 == NCells-1) and (switch_bigger_area==0):
            break
    
    return cellX, cellY, cellVx, cellVy
#################### FUNCTIONS ####################

for i in range(1):

    try:
        cellType    = np.loadtxt('init/Type_init.txt', delimiter=',', dtype=int)
        cellPhi     = np.loadtxt('init/Phi_init.txt', delimiter=',', dtype=float)
        cellR       = np.loadtxt('init/R_init.txt', delimiter=',', dtype=float)
        cellArea = np.pi * cellR**2
        cellState   = np.loadtxt('init/State_init.txt', delimiter=',', dtype=int)
        cellFitness = np.loadtxt('init/Fit_init.txt', delimiter=',', dtype=float)
        
        switch_PhiFR_load = 1
    except:
        switch_PhiFR_load = 0
    
    try:
        os.makedirs("initialize_steps", exist_ok=True)
        os.makedirs("init", exist_ok=True)
    except:
        pass
    
    
    
    
    
    ##### reading params #################################
    filename = 'params.csv'
    variables = read_custom_csv(filename)
    Init_numbers  = np.loadtxt('Init_numbers.csv', delimiter=',', dtype=int)
    NCells = np.sum(Init_numbers)
    if variables['NTypes']>2:
        print("###########################")
        print("Code does not work for NTypes>2 currently!")
        print("###########################")
        sys.exit()
    ##### reading params #################################
    
    if switch_PhiFR_load == 0:
        cellType, cellPhi, cellR, cellArea, cellState, cellFitness = PhiFR_init()
        
    cellX, cellY, cellVx, cellVy = XYVxVy_init()
    
    
    np.savetxt("init"+"/"+"X_init.txt", cellX, fmt='%.6f', delimiter=',')
    np.savetxt("init"+"/"+"Y_init.txt", cellY, fmt='%.6f', delimiter=',')
    np.savetxt("init"+"/"+"Vx_init.txt", cellVx, fmt='%.6f', delimiter=',')
    np.savetxt("init"+"/"+"Vy_init.txt", cellVy, fmt='%.6f', delimiter=',')
    
    if switch_PhiFR_load==0:
        np.savetxt("init"+"/"+"Type_init.txt", cellType, fmt='%d', delimiter=',')
        np.savetxt("init"+"/"+"Phi_init.txt", cellPhi, fmt='%.6f', delimiter=',')
        np.savetxt("init"+"/"+"R_init.txt", cellR, fmt='%.6f', delimiter=',')
        np.savetxt("init"+"/"+"State_init.txt", cellState, fmt='%d', delimiter=',')
        np.savetxt("init"+"/"+"Fit_init.txt", cellFitness, fmt='%.6f', delimiter=',')
    
    np.savetxt("init"+"/"+"Theta_init.txt", 0.0*cellX, fmt='%.6f', delimiter=',')
    
    print("i = "+str(i)+" : done!")

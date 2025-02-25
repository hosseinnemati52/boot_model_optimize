#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 15:34:50 2024

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
from scipy.stats import ks_2samp

CYCLING_STATE =   (1)
G1_ARR_STATE =    (-1)
G0_STATE =        (-2)
DIFF_STATE =      (-3)
APOP_STATE =      (-4)
WIPED_STATE =     (-5)
CA_CELL_TYPE =    (1)
WT_CELL_TYPE =    (0)
NULL_CELL_TYPE =  (-1)

def KS_test(list_of_data, m_checking, p_val_thresh):
    
    checking_list = list_of_data[-m_checking:]
    
    p_val_matrix = np.ones((m_checking, m_checking))
    
    for i in range(m_checking):
        data_i = checking_list[i]
        for j in range(i+1, m_checking):
            data_j = checking_list[j]
            
            # plt.figure()
            # plt.hist(data_i, bins=30, cumulative=True, alpha=0.5, color='blue', density=False)
            # plt.hist(data_j, bins=30, cumulative=True, alpha=0.5, color='red', density=False)
            # plt.show()
            
            
            statistic, p_value = ks_2samp(data_i, data_j)
            
            p_val_matrix[i,j] = p_value
            p_val_matrix[j,i] = p_value
            
    
    result = int( (np.min(p_val_matrix) > p_val_thresh) )
    
    return result



initStepsFolderName = "initialize_steps"

with open("init_steps_data.txt", "r") as init_steps_data:
    
    while(1):
        
        line = init_steps_data.readline()
        
        if line == '':
            break
        
        line_split = line.split()
        
        if line_split[0]=='Dt_mr(h):':
            Dt_mr = float(line_split[1])
        elif line_split[0]=='Dt_eq(h):':
            Dt_eq = float(line_split[1])
        elif line_split[0]=='n_sampling:':
            n_sampling = int(float(line_split[1]))
        elif line_split[0]=='m_checking:':
            m_checking = int(float(line_split[1]))
        
    init_steps_data.close()
    
    
    
    


init_numbers = np.loadtxt("Init_numbers.csv", delimiter=',', dtype=int)
# TypeAllData = []
# StateAllData = []
# PhiAllData = []
# FitnessAllData = []

WT_alive_phi_data_sequence = []
WT_alive_fit_data_sequence = []
C_alive_phi_data_sequence  = []
C_alive_fit_data_sequence  = []



stepIndex = -1
while(1):
    stepIndex += 1
    try:
        cellType  = np.loadtxt(initStepsFolderName+'/Type_'+str(stepIndex)+'.txt', delimiter=',', dtype=int)
        cellState = np.loadtxt(initStepsFolderName+'/State_'+str(stepIndex)+'.txt', delimiter=',', dtype=int)
        cellPhi   = np.loadtxt(initStepsFolderName+'/Phi_'+str(stepIndex)+'.txt', delimiter=',', dtype=float)
        cellFitness   = np.loadtxt(initStepsFolderName+'/Fit_'+str(stepIndex)+'.txt', delimiter=',', dtype=float)
        
        WT_alive_phi_data = []
        WT_alive_fit_data = []
        C_alive_phi_data = []
        C_alive_fit_data = []
        
        WT_alive_phi_data.clear()
        WT_alive_fit_data.clear()
        C_alive_phi_data.clear()
        C_alive_fit_data.clear()
        
        WT_alive_phi_data = []
        WT_alive_fit_data = []
        C_alive_phi_data = []
        C_alive_fit_data = []
        
        for i in range(len(cellType)):
            if (cellType[i]==WT_CELL_TYPE) and (cellState[i] > APOP_STATE):
                WT_alive_phi_data.append(cellPhi[i])
                WT_alive_fit_data.append(cellFitness[i,0])
            elif (cellType[i]==CA_CELL_TYPE) and (cellState[i] > APOP_STATE):
                C_alive_phi_data.append(cellPhi[i])
                C_alive_fit_data.append(cellFitness[i,0])
        
        WT_alive_phi_data = np.array(WT_alive_phi_data)
        WT_alive_fit_data = np.array(WT_alive_fit_data)
        C_alive_phi_data =  np.array(C_alive_phi_data)
        C_alive_fit_data =  np.array(C_alive_fit_data)
        
        WT_alive_phi_data_sequence.append(WT_alive_phi_data)
        WT_alive_fit_data_sequence.append(WT_alive_fit_data)
        C_alive_phi_data_sequence.append(C_alive_phi_data)
        C_alive_fit_data_sequence.append(C_alive_fit_data)
        
    except FileNotFoundError:
        break

    
KS_cond = np.array([0,0,0,0], dtype=int)

if (stepIndex +1 >= n_sampling):
    
    p_val_thresh = 0.05
    
    if init_numbers[WT_CELL_TYPE] > 0:
        KS_cond[0] = KS_test(WT_alive_phi_data_sequence, m_checking, p_val_thresh)
        KS_cond[1] = KS_test(WT_alive_fit_data_sequence, m_checking, p_val_thresh)
    else:
        KS_cond[0] = 1
        KS_cond[1] = 1
        
    if init_numbers[CA_CELL_TYPE] > 0:
        KS_cond[2] = KS_test(C_alive_phi_data_sequence, m_checking, p_val_thresh)
        KS_cond[3] = KS_test(C_alive_fit_data_sequence, m_checking, p_val_thresh)
    else:
        KS_cond[2] = 1
        KS_cond[3] = 1


number_cond = bool(stepIndex +1 >= n_sampling)

eq_cond = (np.product(KS_cond)) and (number_cond)

np.savetxt("eq_condition.txt", [int(eq_cond)], fmt="%d")
    


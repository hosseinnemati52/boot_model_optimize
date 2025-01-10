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

def KS_test():
    
    return 1



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
    
    

TypeAllData = []
StateAllData = []
PhiAllData = []
FitnessAllData = []

stepIndex = -1
while(1):
    stepIndex += 1
    try:
        cellType  = np.loadtxt(initStepsFolderName+'/Type_'+str(stepIndex)+'.txt', delimiter=',', dtype=int)
        cellState = np.loadtxt(initStepsFolderName+'/State_'+str(stepIndex)+'.txt', delimiter=',', dtype=int)
        cellPhi   = np.loadtxt(initStepsFolderName+'/Phi_'+str(stepIndex)+'.txt', delimiter=',', dtype=float)
        cellFitness   = np.loadtxt(initStepsFolderName+'/Fit_'+str(stepIndex)+'.txt', delimiter=',', dtype=float)
    except FileNotFoundError:
        break
    
    TypeAllData.append(cellType)
    StateAllData.append(cellState)
    PhiAllData.append(cellPhi)
    FitnessAllData.append(cellFitness)
    
KS_cond = KS_test()
number_cond = bool(stepIndex -1 >= n_sampling)

eq_cond = (KS_cond) and (number_cond)
np.savetxt("eq_condition.txt", [int(eq_cond)], fmt="%d")
    


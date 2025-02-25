#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 16:25:17 2024

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


N_runs = np.loadtxt('N_runs.csv', delimiter=',', dtype=int)

with open("compos.txt", "r") as compos:
    
    key = compos.readline()[:-1]
    
    if key=='mix':
        a = 0.0
        b = 1.0
    elif key=='WT':
        a = 1.0
        b = 1.0
    elif key=='C':
        a = 0.0
        b = 0.0
    else:
        print("############################")
        print("Wrong compos key!")
        print("############################")
        sys.exit()
    
    compos.close()

WT_init_frac_interval = np.array([a, b])

eps = 0.0001

max_number= 100



if np.mean(WT_init_frac_interval) > (1.0 - eps):
    # pure WT
    file_name = 'sample_banks/WT_sample_bank.csv'
    sample_bank = np.loadtxt(file_name, delimiter=',', dtype=int)
    sample_bank_length = len(sample_bank)
    
    all_indices = np.arange(sample_bank_length)
    
    while 1:
        
        Init_numbers_array = np.zeros((2, N_runs), dtype=int)
        
        chosen_indices = np.random.choice(all_indices, N_runs, replace=False)
        chosen_elements = np.array([sample_bank[i] for i in chosen_indices])
        #init_num_cells = 20*np.ones(N_runs)
        
        Init_numbers_array[0,:] = chosen_elements.copy()
        Init_numbers_array[1,:] = 0 * chosen_elements.copy()
        
        if np.max(Init_numbers_array) <= max_number:
            break
        
    
elif np.mean(WT_init_frac_interval) < eps:
    # pure Cancer
    file_name = 'sample_banks/C_sample_bank.csv'
    sample_bank = np.loadtxt(file_name, delimiter=',', dtype=int)
    sample_bank_length = len(sample_bank)
    
    all_indices = np.arange(sample_bank_length)
    
    while 1:
        
        Init_numbers_array = np.zeros((2, N_runs), dtype=int)
        
        chosen_indices = np.random.choice(all_indices, N_runs, replace=False)
        chosen_elements = np.array([sample_bank[i] for i in chosen_indices])
        #init_num_cells = 20*np.ones(N_runs)
        
        Init_numbers_array[0,:] = 0 * chosen_elements.copy()
        Init_numbers_array[1,:] = chosen_elements.copy()
        
        if np.max(Init_numbers_array) <= max_number:
            break
    
    
else:
    # mixed
    file_name = 'sample_banks/mixed_sample_bank.csv'
    sample_bank = np.loadtxt(file_name, delimiter=',', dtype=int)
    sample_bank_length = np.shape(sample_bank)[0]
    
    all_indices = np.arange(sample_bank_length)
    
    while 1:
        
        Init_numbers_array = np.zeros((2, N_runs), dtype=int)
        
        chosen_indices = np.random.choice(all_indices, N_runs, replace=False)
        
        init_num_cells_WT = []
        init_num_cells_C  = []
        
        init_num_cells_WT.clear()
        init_num_cells_C.clear()
        
        init_num_cells_WT = []
        init_num_cells_C  = []
        
        for i in chosen_indices:
            init_num_cells_WT.append(sample_bank[i,0])
            init_num_cells_C.append(sample_bank[i,1])
        
        init_num_cells_WT=np.array(init_num_cells_WT)
        init_num_cells_C=np.array(init_num_cells_C)
        
        Init_numbers_array[0, :] = init_num_cells_WT
        Init_numbers_array[1, :] = init_num_cells_C
        
        if np.max(Init_numbers_array) <= max_number:
            break

np.savetxt("Init_numbers_array.csv", Init_numbers_array, delimiter=',', fmt='%d')


# plt.figure()
# plt.hist(Init_numbers_array[0,:], bins=30, density=True)

# plt.figure()
# plt.hist(Init_numbers_array[1,:], bins=30, density=True)

for runC in range(N_runs):
    folderName = "run_"+str(runC+1)
    
    Init_numbers_file = Init_numbers_array[:,runC]
    
    np.savetxt(folderName +"/"+"Init_numbers.csv", Init_numbers_file, delimiter=',', fmt='%d')

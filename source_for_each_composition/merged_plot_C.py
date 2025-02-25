#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 08:11:29 2024

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




exp_array_pure = np.loadtxt("exp_data"+"/"+"overal_C_pure.csv", delimiter=',')
x = exp_array_pure[0,:]
y = exp_array_pure[1,:]
y_err = exp_array_pure[2,:]
plt.scatter(x, y, label='pure C (exp)', zorder=10)
plt.errorbar(x, y, yerr=y_err/np.sqrt(50), fmt='o', zorder=10)

sim_array_pure = np.loadtxt("overal_pp_pure"+"/"+"norm_C_ov_pl.txt", delimiter=',')
x = sim_array_pure[0,:]
y = sim_array_pure[1,:]
y_err = sim_array_pure[2,:]
plt.scatter(x, y, label='pure C (sim)', marker='*')
plt.errorbar(x, y, yerr=y_err/np.sqrt(20), fmt='o', marker='*')




exp_array_mix = np.loadtxt("exp_data"+"/"+"C_bar_mix_overal.csv", delimiter=',')
x = exp_array_mix[0,:]
y = exp_array_mix[1,:]
y_err = exp_array_mix[2,:]
plt.scatter(x, y, label='mix C (exp)', zorder=10, color='k')
plt.errorbar(x, y, yerr=y_err/np.sqrt(50), fmt='o', zorder=10, color='k')

sim_array_mix = np.loadtxt("overal_pp_mix"+"/"+"norm_C_ov_pl.txt", delimiter=',')
x = sim_array_mix[0,:]
y = sim_array_mix[1,:]
y_err = sim_array_mix[2,:]
plt.scatter(x, y, label='mix C (sim)', marker='*')
plt.errorbar(x, y, yerr=y_err/np.sqrt(20), fmt='o', marker='*')


plt.xlabel('time(h)', fontsize=15)
plt.ylabel('Normalized Number', fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
# plt.grid()
# plt.yscale("log")
plt.legend(fontsize=15)
plt.tight_layout()

plt.savefig("C_compare.PNG", dpi=300)
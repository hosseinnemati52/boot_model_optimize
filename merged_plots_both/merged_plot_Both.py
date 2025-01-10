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
plt.scatter(x, y, label='pure C (exp)', zorder=10, marker='o', color='g')
plt.errorbar(x, y, yerr=y_err/np.sqrt(50), fmt='o', zorder=10, color='g')


exp_array_pure = np.loadtxt("exp_data"+"/"+"overal_WT_pure.csv", delimiter=',')
x = exp_array_pure[0,:]
y = exp_array_pure[1,:]
y_err = exp_array_pure[2,:]
plt.scatter(x, y, label='pure WT (exp)', zorder=10, color='purple')
# plt.plot(x, y, label='pure WT (exp)', zorder=10)
plt.errorbar(x, y, yerr=y_err/np.sqrt(50), fmt='o', zorder=10, color='purple')


exp_array_mix = np.loadtxt("exp_data"+"/"+"C_bar_mix_overal.csv", delimiter=',')
x = exp_array_mix[0,:]
y = exp_array_mix[1,:]
y_err = exp_array_mix[2,:]
plt.scatter(x, y, label='mix C (exp)', zorder=10, color='g', marker='*')
plt.errorbar(x, y, yerr=y_err/np.sqrt(50), fmt='*', zorder=10, color='g')


exp_array_mix = np.loadtxt("exp_data"+"/"+"WT_bar_mix_overal.csv", delimiter=',')
x = exp_array_mix[0,:]
y = exp_array_mix[1,:]
y_err = exp_array_mix[2,:]
plt.scatter(x, y, label='mix WT (exp)', zorder=10, color='purple', marker='*')
plt.errorbar(x, y, yerr=y_err/np.sqrt(50), fmt='*', zorder=10, color='purple')



sim_array_pure = np.loadtxt("overal_pp_C"+"/"+"norm_C_ov_pl.txt", delimiter=',')
x = sim_array_pure[0,:]
y = sim_array_pure[1,:]
y_err = sim_array_pure[2,:]
# plt.scatter(x, y, label='pure C (sim)', marker='*')
# plt.plot(x, y, label='pure C (sim)', color= 'g', linestyle='dashed')
plt.errorbar(x, y, yerr=y_err/np.sqrt(20), fmt='o', marker='*', markersize=1, linestyle='dashed', label='pure C (sim)')



sim_array_pure = np.loadtxt("overal_pp_WT"+"/"+"norm_WT_ov_pl.txt", delimiter=',')
x = sim_array_pure[0,:]
y = sim_array_pure[1,:]
y_err = sim_array_pure[2,:]
# plt.scatter(x, y, label='pure WT (sim)', marker='*')
# plt.plot(x, y, label='pure WT (sim)')
plt.errorbar(x, y, yerr=y_err/np.sqrt(20), fmt='o', marker='*', markersize=1, label='pure WT (sim)', linestyle='dashed')


sim_array_mix = np.loadtxt("overal_pp_mix"+"/"+"norm_C_ov_pl.txt", delimiter=',')
x = sim_array_mix[0,:]
y = sim_array_mix[1,:]
y_err = sim_array_mix[2,:]
# plt.plot(x, y, label='mix C (sim)')
plt.errorbar(x, y, yerr=y_err/np.sqrt(20), fmt='o', marker='*', markersize=1, label='mix C (sim)', linestyle='dashed')


sim_array_mix = np.loadtxt("overal_pp_mix"+"/"+"norm_WT_ov_pl.txt", delimiter=',')
x = sim_array_mix[0,:]
y = sim_array_mix[1,:]
y_err = sim_array_mix[2,:]
# plt.plot(x, y, label='mix WT (sim)')
plt.errorbar(x, y, yerr=y_err/np.sqrt(20), fmt='o', marker='*', markersize=1 , linestyle='dashed', label='mix WT (sim)')

plt.xlabel('time(h)', fontsize=15)
plt.ylabel('Normalized Number', fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlim((0,71))
# plt.grid()
plt.yscale("log")
# plt.legend(fontsize=15)
plt.legend(loc='upper center', bbox_to_anchor=(0.4, 1.05),
          ncol=2, fancybox=True, shadow=False, fontsize=12)
plt.tight_layout()

plt.savefig("Both_compare.PNG", dpi=300)

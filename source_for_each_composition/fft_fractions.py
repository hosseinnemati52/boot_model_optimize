#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 17:58:05 2024

@author: hossein
"""

import numpy as np
import matplotlib.pyplot as plt


all_data = np.loadtxt("overal_pp/frac_WT_g1_ov_pl.txt", delimiter=',')

t = all_data[0,:]
y = all_data[1,:]


# Assuming you have time (t) and signal (y)
fft_result = np.fft.fft(y)
freqs = np.fft.fftfreq(len(y), d=np.mean(np.diff(t)))
power = np.abs(fft_result)**2

plt.plot(freqs[freqs > 0], power[freqs > 0])  # Only positive frequencies
plt.xlabel("Frequency")
plt.ylabel("Power")
plt.title("Power Spectral Density")
plt.xlim((0,1))
plt.show()
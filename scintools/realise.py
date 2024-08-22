#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 20:02:07 2024

@author: dreardon
"""

import numpy as np
import matplotlib.pyplot as plt

Tspan = 1000  # Number of time samples
N = 1000  # Number of Fourier components
A = 1
alpha = -8/3  # spectral index

t = np.linspace(0, Tspan, Tspan)  # time domain samples
f = np.linspace(1, N, N) / Tspan  # N Fourier coefficients

P = A * f ** alpha

x = np.zeros_like(t)
for i in range(len(f)):
    phi = np.random.rand() * 2*np.pi
    x += np.sqrt(P[i]) * np.sin(2*np.pi*f[i]*t + phi)

plt.scatter(t, x, marker='x')
plt.xlabel('Sample number')
plt.ylabel('Observable value')
plt.show()

fx = np.fft.fft(x)
px = fx * np.conj(fx)

nx = len(px)

plt.plot(px[:nx//2])
plt.ylabel('Power (arb)')
plt.xlabel('Frequency (arb)')
plt.yscale('log')
plt.xscale('log')
# plt.show()


# Interpolate the time series process using a changing Vlos

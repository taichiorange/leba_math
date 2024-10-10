# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 22:22:24 2024

@author: 41377
"""

import numpy as np
import matplotlib.pyplot as plt

Nfft = 64        # fft N

N = 240000


# OFDM
N_sym = int(N/Nfft)
Df = np.zeros((Nfft, N_sym),dtype='complex')

Df[1,:] = np.exp(1j*np.pi/4)

# to time domain
Dt = np.fft.ifft(Df,axis=0)
Dt_snd = np.reshape(Dt,N,order='F')

plt.plot(np.real(Dt_snd[0:500:1]),".--")

print(max(np.real(Dt_snd)))
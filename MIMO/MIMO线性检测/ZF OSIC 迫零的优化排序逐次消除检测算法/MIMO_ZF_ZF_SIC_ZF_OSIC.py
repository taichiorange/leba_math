# -*- coding: utf-8 -*-
"""
Created on Fri May 19 20:32:42 2023

@author: 41377
"""

import numpy as np
import matplotlib.pyplot as plt

from numpy.random import randint 
from numpy.random import randn
from numpy import sqrt

Nt = 2;
Nr = Nt;

N = int(1e6)

SNRs = np.arange(-10,30,2)

BERs_ZF = np.zeros(np.size(SNRs));
BERs_ZF_SIC = np.zeros(np.size(SNRs));
BERs_ZF_OSIC = np.zeros(np.size(SNRs));

x_hat_SIC = np.zeros([Nt,1])
x_hat_OSIC = np.zeros([Nt,1])


for kk in range(0,len(SNRs)):
    snr = SNRs[kk]
    Pnoise = 1/10**(snr/10)
    NerrorZF = 0
    NerrorZF_SIC = 0
    NerrorZF_OSIC = 0
    for ii in range(1,N):
        #c = randi([0,1],Nt,1);
        c = randint(0,2, size=[Nt,1])
        x = 2*c - 1
        H = np.matrix(1/sqrt(Nt) * (randn(Nr,Nt) + 1j*randn(Nr,Nt))/sqrt(2))
        n = sqrt(Pnoise/2) * (randn(Nr,1) + 1j * randn(Nr,1))
        
        y = H@x + n
        
        # zero forcing
        x_hat = np.linalg.inv((H.H@ H)) @ H.H @ y
        x_hat = 2*(x_hat>0)-1;
        NerrorZF = NerrorZF + sum( x != x_hat)
        
        # zero forcing with SIC
        x1 = x_hat[1,0]
        yc = y - x1 * H[:,1]
        temp = H[:,0].H @yc /(H[:,0].H@H[:,0])
        x_hat_SIC[0,0] = 2*(temp>0)-1
        x_hat_SIC[1,0] = x1
        NerrorZF_SIC = NerrorZF_SIC + sum( x != x_hat_SIC)
        
        # zero forcing with optimized SIC
        if H[:,0].H @ H[:,0] > H[:,1].H @ H[:,1]:
            h_1st = H[:,0]
            h_2nd = H[:,1]
            x1 = x_hat[0,0]
            yc = y - x1 * h_1st
            temp = h_2nd.H @yc /(h_2nd.H@h_2nd)
            x_hat_OSIC[1,0] = 2*(temp>0)-1
            x_hat_OSIC[0,0] = x1            
        else:
            h_1st = H[:,1]
            h_2nd = H[:,0]
            x1 = x_hat[1,0]
            yc = y - x1 * h_1st
            temp = h_2nd.H @yc /(h_2nd.H@h_2nd)
            x_hat_OSIC[0,0] = 2*(temp>0)-1
            x_hat_OSIC[1,0] = x1         

        NerrorZF_OSIC = NerrorZF_OSIC + sum( x != x_hat_OSIC)
            
        
        
    BERs_ZF[kk] = NerrorZF/(N*Nt)
    BERs_ZF_SIC[kk] = NerrorZF_SIC/(N*Nt)
    BERs_ZF_OSIC[kk] = NerrorZF_OSIC/(N*Nt)

fig = plt.figure()

ax = fig.add_subplot(1,1,1)
ax.semilogy(SNRs,BERs_ZF,label="ZF")
ax.semilogy(SNRs,BERs_ZF_SIC,label="ZF_SIC")
ax.semilogy(SNRs,BERs_ZF_OSIC,label="ZF_OSIC")
ax.legend()
ax.grid(b=True, which='major', linestyle='-')
ax.grid(b=True, which='minor', linestyle='--')
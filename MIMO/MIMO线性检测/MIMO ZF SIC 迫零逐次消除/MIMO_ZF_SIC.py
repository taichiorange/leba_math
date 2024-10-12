# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from numpy.random import randint 
from numpy.random import randn
from numpy import sqrt

Nt = 8;
Nr = Nt;

N = int(1e4)

SNRs = np.arange(-10,30,2)

BERs_ZF = np.zeros(np.size(SNRs));
BERs_ZF_SIC = np.zeros(np.size(SNRs));

x_hat_SIC = np.zeros([Nt,1])



for kk in range(0,len(SNRs)):
    snr = SNRs[kk]
    Pnoise = 1/10**(snr/10)
    NerrorZF = 0
    NerrorZF_SIC = 0

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
        for jj in range(0,Nt):
            x_hat = np.linalg.inv((H.H@ H)) @ H.H @ y
            x_hat = 2*(x_hat>0)-1;
            
            x_hat_SIC[Nt-jj-1] = x_hat[-1,0]
            y = y - H[:,Nt-jj-1] * x_hat_SIC[Nt-jj-1]
            H = H[:,0:Nt-jj-1]
        NerrorZF_SIC = NerrorZF_SIC + sum( x != x_hat_SIC)
        
    BERs_ZF[kk] = NerrorZF/(N*Nt)
    BERs_ZF_SIC[kk] = NerrorZF_SIC/(N*Nt)


fig = plt.figure()

ax = fig.add_subplot(1,1,1)
ax.semilogy(SNRs,BERs_ZF,label="ZF")
ax.semilogy(SNRs,BERs_ZF_SIC,label="ZF_SIC")
ax.legend()
ax.grid(b=True, which='major', linestyle='-')
ax.grid(b=True, which='minor', linestyle='--')


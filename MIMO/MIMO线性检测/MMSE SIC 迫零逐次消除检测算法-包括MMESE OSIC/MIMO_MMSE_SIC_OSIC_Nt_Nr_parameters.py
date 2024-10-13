# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from numpy.random import randint 
from numpy.random import randn
from numpy import sqrt

Nt = 4;
Nr = Nt;

N = int(1e5)

SNRs = np.arange(-10,30,1)

BERs_MMSE = np.zeros(np.size(SNRs));
BERs_MMSE_SIC = np.zeros(np.size(SNRs));
BERs_MMSE_OSIC = np.zeros(np.size(SNRs));

x_hat_SIC = np.zeros([Nt,1])
x_hat_OSIC = np.zeros([Nt,1])

for kk in range(0,len(SNRs)):
    snr = SNRs[kk]
    Pnoise = 1/10**(snr/10)
    NerrorMMSE = 0
    NerrorMMSE_SIC = 0
    NerrorMMSE_OSIC = 0

    for ii in range(1,N):
        #c = randi([0,1],Nt,1);
        c = randint(0,2, size=[Nt,1])
        x = 2*c - 1
        H = np.matrix(1/sqrt(Nt) * (randn(Nr,Nt) + 1j*randn(Nr,Nt))/sqrt(2))
        n = sqrt(Pnoise/2) * (randn(Nr,1) + 1j * randn(Nr,1))
        
        y = H@x + n
        
        # MMSE
        x_hat = np.linalg.inv((H.H@ H)+Pnoise * np.eye(Nt)) @ H.H @ y
        x_hat = 2*(x_hat>0)-1;
        NerrorMMSE = NerrorMMSE + sum( x != x_hat)
        
        H_save = H
        y_save = y
        
        # MMSE with SIC
        for jj in range(0,Nt):
            x_hat = np.linalg.inv((H.H@ H)+Pnoise * np.eye(Nt-jj)) @ H.H @ y
            x_hat = 2*(x_hat>0)-1;
            
            x_hat_SIC[Nt-jj-1] = x_hat[-1,0]
            y = y - H[:,Nt-jj-1] * x_hat_SIC[Nt-jj-1]
            H = H[:,0:Nt-jj-1]
        NerrorMMSE_SIC = NerrorMMSE_SIC + sum( x != x_hat_SIC)
        
        
        y=y_save
        # zero forcing with optimized order
        ## to sorting the H's column
        order_ind = np.argsort(sum(np.abs(H_save)**2))
        order_ind = order_ind.tolist()
        order_ind = order_ind[0]
        for jj in range(0,Nt):
            H = H_save[:,order_ind[0:Nt-jj]]
            x_hat = np.linalg.inv((H.H@ H)+Pnoise * np.eye(Nt-jj)) @ H.H @ y
            x_hat = 2*(x_hat>0)-1;
            
            ind = order_ind[Nt-jj-1]
            x_hat_OSIC[ind] = x_hat[-1,0]
            y = y - H[:,-1] * x_hat_OSIC[ind]
 
        NerrorMMSE_OSIC = NerrorMMSE_OSIC + sum( x != x_hat_OSIC)
        
        
    BERs_MMSE[kk] = NerrorMMSE/(N*Nt)
    BERs_MMSE_SIC[kk] = NerrorMMSE_SIC/(N*Nt)
    BERs_MMSE_OSIC[kk] = NerrorMMSE_OSIC/(N*Nt)

fig = plt.figure()

ax = fig.add_subplot(1,1,1)
ax.semilogy(SNRs,BERs_MMSE,label="MMSE")
ax.semilogy(SNRs,BERs_MMSE_SIC,label="MMSE_SIC")
ax.semilogy(SNRs,BERs_MMSE_OSIC,label="MMSE_OSIC")
ax.legend()
ax.grid(b=True, which='major', linestyle='-')
ax.grid(b=True, which='minor', linestyle='--')



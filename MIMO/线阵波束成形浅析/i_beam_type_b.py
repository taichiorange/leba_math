# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 12:43:13 2024

@author: 41377
"""

# 画不同的 beam 类型
# k 是小数
import numpy as np
import matplotlib.pyplot as plt

N = 8
dVsLambda = 0.5
k = 7.5
n = np.arange(0,N,1)
 
phaseRotateVector = np.exp(1j*2*k*np.pi/N*n)
thetaAll = np.arange(0,2*np.pi,0.01)
beamAmp = []
for theta in thetaAll:
    ePhi = np.exp(1j*2*np.pi*dVsLambda*np.cos(theta)*n)
    BeamAmpForOneTheta = np.dot(phaseRotateVector,ePhi)
    beamAmp.append(BeamAmpForOneTheta)
beamAmp = np.array(beamAmp)/N

fig = plt.figure(figsize=(10,10))
ax = plt.subplot(111, polar=True)
ax.plot(thetaAll, abs(beamAmp),"r")
ax.grid(True)
ax.set_theta_offset(-1*np.pi/2) 
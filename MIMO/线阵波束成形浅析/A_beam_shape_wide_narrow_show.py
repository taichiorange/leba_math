# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 11:45:30 2024

@author: 41377
"""

import  numpy as np
from    matplotlib import pyplot as plt

N = 8      #天线数量

theta = np.arange(0.000001,2*np.pi-0.0000001,0.01)

psi = np.pi * np.cos(theta)

r = np.abs(np.sin(N * psi/2)/np.sin(psi/2))/N

plt.figure()
plt.polar(theta,r)
plt.show()


N = 16      #天线数量

theta = np.arange(0.000001,2*np.pi-0.0000001,0.01)

psi = np.pi * np.cos(theta)

r = np.abs(np.sin(N * psi/2)/np.sin(psi/2))/N

plt.figure()
plt.polar(theta,r)
plt.show()
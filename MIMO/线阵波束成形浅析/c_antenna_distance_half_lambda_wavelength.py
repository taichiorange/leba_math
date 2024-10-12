# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 12:21:55 2024

@author: 41377
"""

import  numpy as np
from    matplotlib import pyplot as plt

N = 32      #天线数量


psi = np.arange(-2*np.pi,2*np.pi-0.0000001,0.01)


r = np.abs(np.sin(N * psi/2)/np.sin(psi/2))/N

plt.figure()
plt.plot(psi/np.pi,r)
plt.grid()
plt.show()
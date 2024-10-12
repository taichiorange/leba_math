# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 12:27:18 2024

@author: 41377
"""

import  numpy as np
from    matplotlib import pyplot as plt

N = 32      #天线数量

theta = np.arange(-np.pi,np.pi-0.0000001,0.01)

d_vs_lambda = 0.01   # 请修改这个值，即使用不同的  d/lambda

psi = 2 * np.pi * d_vs_lambda * np.cos(theta)

r = np.abs(np.sin(N * psi/2)/np.sin(psi/2))/N

plt.figure()
plt.polar(theta,r)
plt.show()

plt.figure()
plt.plot(theta/np.pi,r)
plt.ylim([0,1.1])
plt.show()
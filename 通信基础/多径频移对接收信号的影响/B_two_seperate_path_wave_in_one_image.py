# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 22:07:56 2024

@author: 41377
"""

import matplotlib.pyplot as plt
import numpy as np

PI = np.pi

t = np.arange(0,0.1,0.0001)

fc = 100

fd1 = 5
fd2 = 7

s1 = np.cos(2*PI*(fc-fd1)*t)
s2 = np.cos(2*PI*(fc-fd2)*t)

r = s1 + s2

fig,ax1 = plt.subplots(1,1)
ax1.plot(t,s1,t,s2)
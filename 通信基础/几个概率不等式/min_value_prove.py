# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 21:10:08 2024

@author: 41377
"""

import numpy as np
import matplotlib . pyplot as plt
n = 10
m = 7
p = 1/2
t = np.arange (0 ,2 ,0.01)
y = n * np.log (p*np.exp ( t)+1-p) - t*m
ey = np.exp (y)
plt.plot ( t , ey )
plt.grid ()
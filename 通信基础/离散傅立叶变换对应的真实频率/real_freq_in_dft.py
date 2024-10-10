# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 20:50:51 2024

@author: 41377
"""

import numpy as np
from matplotlib import pyplot as plt
N = 128
fs = 8192
dt = 1/fs
k = N-10 # 与 −10 频 率 是 相 同 的
rt = 10
tc = np.arange (0 ,32/8192 , dt ) # coarse time
tf = np.arange (0 ,32/8192 , dt /10) # fine time
dc = np.sin(2*np.pi*k*fs/N*tc) # data on coarse time
df = np.sin(2*np.pi*k*fs/N*tf) # data on fine time
kk = -10
da = np.sin (2*np.pi*kk*fs/N*tc) # data on actual frequency
plt.figure ()
plt.subplot (2 ,1 ,1) , plt . plot ( tc , dc ,"r.-" , tf , df ,"g--")
plt.subplot (2 ,1 ,2) , plt . plot ( tc , da ,"r.-")
plt.show ()
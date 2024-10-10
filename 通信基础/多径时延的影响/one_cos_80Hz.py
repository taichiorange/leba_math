# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 21:53:18 2024

@author: 41377
"""
# 按照同样方法，另外一个频率是 80 Hz，执行下面的 python 代码：

import matplotlib.pyplot as plt
import numpy as np

fc1 = 80   # 频率1, 80Hz

# 时间点，采样的时间点
t = np.arange(0,0.1,0.0001)

s_fc1 = np.cos(2*np.pi*fc1*t) 

fig,ax1= plt.subplots(1,1)
ax1.plot(t,s_fc1,"g.--")

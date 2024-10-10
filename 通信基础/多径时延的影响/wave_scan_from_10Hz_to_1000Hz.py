# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 21:58:18 2024

@author: 41377
"""

import matplotlib.pyplot as plt
import numpy as np

#下面，我们用同样两个时延的路径，把频率从 10 Hz 到 1000Hz 扫一遍，看不同频率下两路时延信号叠加后的最大幅度的变化：

# 生成频点：10 Hz 到 1000Hz，
fc = np.arange(10,1000,1)

a=[]
tau1 = 0.01  # 路径 1 的时延
tau2 = 0.015 # 路径 2 的时延

# 时间点，采样的时间点
t = np.arange(0,0.1,0.0001)

# 对每个频点，计算其最高幅度值
for f in fc:
    s = np.cos(2*np.pi*f*t)   # 发送的原始信号

    s1 = np.cos(2*np.pi*f*(t+tau1))  #路径 1 时延 后的信号
    s2 = np.cos(2*np.pi*f*(t+tau2))  #路径 2 时延 后的信号
    
    r = s1 + s2   # 两路信号叠加后收到的信号
    
    a.append(max(abs(r)))   #找到最大幅度值

fig,ax1 = plt.subplots(1,1)
ax1.plot(fc,a,".--")
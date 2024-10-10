# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 21:50:41 2024

@author: 41377
"""

# 这个频率的信号，有两个路径传输到接收端，每个路径的时延分别是 0.01秒和 0.015秒，经过两路时延后收到的信号，分别显示在下图中：
# 红色：是原始发出的信号
# 绿色：是时延 0.01秒那一路的信号
# 绿色：是时延 0.015秒那一路的信号
# 黄色：是两路不同时延的信号，在接收端叠加后的结果
# 代码如下：

import matplotlib.pyplot as plt
import numpy as np

fc1 = 20   # 频率1, 20Hz

tau1 = 0.01  # 路径 1 的时延
tau2 = 0.015 # 路径 2 的时延

# 时间点，采样的时间点
t = np.arange(0,0.1,0.0001)

s_fc1 = np.cos(2*np.pi*fc1*t) 

s1 = np.cos(2*np.pi*fc1*(t+tau1))  #路径 1 时延 后的信号
s2 = np.cos(2*np.pi*fc1*(t+tau2))  #路径 2 时延 后的信号

r_fc1 = s1 + s2   # 发射的信号是频率1时，两路信号叠加后收到的信号

fig,ax1 = plt.subplots(1,1)
ax1.plot(t,s_fc1,"r.--",t,s1,"g.--",t,s2,"b.--",t,r_fc1,"y.--")
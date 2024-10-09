# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

f = 100     # Hz
df = 10     # 频率偏差


# 时间点，采样的时间点
t = np.arange(0,0.04,0.0001)

s_f = np.cos(2*np.pi*f*t) 

s_df = np.cos(2*np.pi*(f+df)*t)




fig = plt.figure()

ax1 = fig.add_subplot(2,2,1)
ax2 = fig.add_subplot(2,2,2)

ax3 = fig.add_subplot(2,1,2)

ax1.plot(t,s_f,"g-")
ax2.plot(t,s_df,"r--")

ax3.plot(t,s_f,"g-",t,s_df,"r--")



s_f = np.sin(2*np.pi*f*t) 

s_df = np.sin(2*np.pi*(f+df)*t)




fig = plt.figure(2)

ax1 = fig.add_subplot(2,2,1)
ax2 = fig.add_subplot(2,2,2)

ax3 = fig.add_subplot(2,1,2)

ax1.plot(t,s_f,"g-")
ax2.plot(t,s_df,"r--")

ax3.plot(t,s_f,"g-",t,s_df,"r--")

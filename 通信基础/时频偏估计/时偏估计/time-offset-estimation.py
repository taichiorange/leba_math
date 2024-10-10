# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

f1 = 100     # Hz
f2 = 200


# 时间点，采样的时间点
t = np.arange(0,0.04,0.0001)

s1 = np.cos(2*np.pi*f1*t) 

s2 = np.cos(2*np.pi*f2*t) 



fig = plt.figure()

ax1 = fig.add_subplot(2,2,1)
ax2 = fig.add_subplot(2,2,2)

ax3 = fig.add_subplot(2,1,2)

ax1.plot(t,s1,"g-")
ax1.axvline(x=0.01,color="r")
ax1.axvline(x=0.02,color="r")
ax1.axvline(x=0.03,color="r")

ax2.plot(t,s2,"r--")
ax2.axvline(x=0.01,color="g")
ax2.axvline(x=0.02,color="g")
ax2.axvline(x=0.03,color="g")

ax3.plot(t,s1,"g-",t,s2,"r--")
ax3.axvline(x=0.01,color="b")
ax3.axvline(x=0.02,color="b")
ax3.axvline(x=0.03,color="b")



dt = 0.002
fig = plt.figure()

ax1 = fig.add_subplot(2,2,1)
ax2 = fig.add_subplot(2,2,2)

ax3 = fig.add_subplot(2,1,2)

ax1.plot(t,s1,"g-")
ax1.axvline(x=0.01+dt,color="r")
ax1.axvline(x=0.02+dt,color="r")
ax1.axvline(x=0.03+dt,color="r")

ax2.plot(t,s2,"r--")
ax2.axvline(x=0.01+dt,color="g")
ax2.axvline(x=0.02+dt,color="g")
ax2.axvline(x=0.03+dt,color="g")

ax3.plot(t,s1,"g-",t,s2,"r--")
ax3.axvline(x=0.01+dt,color="b")
ax3.axvline(x=0.02+dt,color="b")
ax3.axvline(x=0.03+dt,color="b")

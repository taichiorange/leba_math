# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt


x = np.arange(-5,5,0.01)
y = 1/np.sqrt(2 * np.pi) * np.exp(-x**2/2)

# 为了画 Q(1.2) 的区域
x1 = np.arange(1.2,5,0.01)
y1 = 1/np.sqrt(2 * np.pi) * np.exp(-x1**2/2)

fig = plt.figure()

ax = fig.add_subplot(1,1,1)

ax.plot(x,y)
# 画 x 和 y 坐标轴
ax.axvline(x=0,color="b")
ax.axhline(y=0,color="b")
## 画坐标轴的箭头
ax.plot(5,0,">")
ax.plot(0,np.max(y)+0.1,"^")

# 画填充的区域, Q 函数的区域 
ax.fill_between(x1, y1,color="r")


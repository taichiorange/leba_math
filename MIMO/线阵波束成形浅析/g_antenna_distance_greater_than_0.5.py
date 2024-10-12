# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 12:36:27 2024

@author: 41377
"""

import  numpy as np
from    matplotlib import pyplot as plt

FigSize=[10,10]

CYCLES = 2
rotateAng = np.pi/3
theta =  np.arange(0,2*np.pi * CYCLES,0.2)
dVsLambda = 0.7

ySin = np.sin(theta)

# rotate the drawing, clockwise by rotateAng
rotatePictureAngle = -(np.pi/2-rotateAng)
x1 = theta * np.cos(rotatePictureAngle) - ySin * np.sin(rotatePictureAngle)
y1 = theta * np.sin(rotatePictureAngle) + ySin * np.cos(rotatePictureAngle)

y2 = y1 + 2 * np.pi * dVsLambda

plt.figure(figsize=FigSize)
plt.plot(x1,y1)
plt.plot(x1,y2)

# 2*np.pi * CYCLES*np.cos(rotatePictureAngle): the x-ax is smaller after rotating
thetaProj =  np.arange(0,2*np.pi * CYCLES*np.cos(rotatePictureAngle),0.2)
plt.plot(thetaProj ,np.tan(rotatePictureAngle)*thetaProj,"g")
plt.plot(thetaProj,np.tan(rotatePictureAngle)*thetaProj + 2 * np.pi * dVsLambda,"g")
plt.axis('equal')
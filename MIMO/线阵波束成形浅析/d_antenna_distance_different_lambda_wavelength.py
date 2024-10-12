# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 12:24:59 2024

@author: 41377
"""

import  numpy as np
from    matplotlib import pyplot as plt

N = 8      #天线数量

theta = np.arange(0.000001,2*np.pi-0.0000001,0.01)

d_vs_lambda = 0.5

psi = 2 * np.pi * d_vs_lambda * np.cos(theta)

r = np.abs(np.sin(N * psi/2)/np.sin(psi/2))/N

plt.figure()
plt.polar(theta,r)
plt.title("0.5")
plt.show()


d_vs_lambda = 0.6

psi = 2 * np.pi * d_vs_lambda * np.cos(theta)

r = np.abs(np.sin(N * psi/2)/np.sin(psi/2))/N

plt.figure()
plt.polar(theta,r)
plt.title("0.6")
plt.show()

d_vs_lambda = 0.7

psi = 2 * np.pi * d_vs_lambda * np.cos(theta)

r = np.abs(np.sin(N * psi/2)/np.sin(psi/2))/N

plt.figure()
plt.polar(theta,r)
plt.title("0.7")
plt.show()



d_vs_lambda = 0.8

psi = 2 * np.pi * d_vs_lambda * np.cos(theta)

r = np.abs(np.sin(N * psi/2)/np.sin(psi/2))/N

plt.figure()
plt.polar(theta,r)
plt.title("0.8")
plt.show()



d_vs_lambda = 0.9

psi = 2 * np.pi * d_vs_lambda * np.cos(theta)

r = np.abs(np.sin(N * psi/2)/np.sin(psi/2))/N

plt.figure()
plt.polar(theta,r)
plt.title("0.9")
plt.show()


d_vs_lambda = 1.0

psi = 2 * np.pi * d_vs_lambda * np.cos(theta)

r = np.abs(np.sin(N * psi/2)/np.sin(psi/2))/N

plt.figure()
plt.polar(theta,r)
plt.title("1.0")
plt.show()
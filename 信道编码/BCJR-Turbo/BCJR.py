# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 11:44:55 2024

@author: 41377
"""

from matplotlib import pyplot as plt
import numpy as np

r = [ 2.53008, 0.731636,  -0.523916, 1.93052,  -0.793262, 0.307327,   - 1.24029, 0.784426,
1.83461, -0.968171,  -0.433259, 1.26344,   1.31717, 0.995695,  -1.50301, 2.04413,
1.60015, -1.15293,  0.108878, -1.57889]

SQUARE_sigma = 0.45

alpha = []

alpha_0 = 1
alpha_1 = 0
alpha_2 = 0
alpha_3 = 0

print("-----time = 0, alpha 0  ------------------------")
print(alpha_0)
print(alpha_1)
print(alpha_2)
print(alpha_3)


alpha.append([alpha_0,alpha_1,alpha_2,alpha_3])

for k in range(0,9):

    ra = r[2*k+0]
    rb = r[2*k+1]

    lambda_pie_00 = np.exp(-1/(2*SQUARE_sigma)*((ra+1)**2 + (rb+1)**2))   # v=00,  -1, -1
    lambda_pie_02 = np.exp(-1/(2*SQUARE_sigma)*((ra-1)**2 + (rb-1)**2))   # v=11,  +1, +1

    lambda_pie_10 = np.exp(-1/(2*SQUARE_sigma)*((ra-1)**2 + (rb+1)**2))   # v=10,  +1, -1
    lambda_pie_12 = np.exp(-1/(2*SQUARE_sigma)*((ra+1)**2 + (rb-1)**2))   # v=01,  -1, +1

    lambda_pie_21 = np.exp(-1/(2*SQUARE_sigma)*((ra+1)**2 + (rb+1)**2))   # v=00,  -1, -1
    lambda_pie_23 = np.exp(-1/(2*SQUARE_sigma)*((ra-1)**2 + (rb-1)**2))   # v=11,  +1, +1

    lambda_pie_31 = np.exp(-1/(2*SQUARE_sigma)*((ra-1)**2 + (rb+1)**2))   # v=10,  +1, -1
    lambda_pie_33 = np.exp(-1/(2*SQUARE_sigma)*((ra+1)**2 + (rb-1)**2))   # v=01,  -1, +1

    alpha_0_tmp = alpha_0 *lambda_pie_00 + alpha_1 * lambda_pie_10
    alpha_1_tmp = alpha_2 *lambda_pie_21 + alpha_3 * lambda_pie_31
    alpha_2_tmp = alpha_0 *lambda_pie_02 + alpha_1 * lambda_pie_12
    alpha_3_tmp = alpha_2 *lambda_pie_23 + alpha_3 * lambda_pie_33

    alpha_sum = alpha_0_tmp + alpha_1_tmp + alpha_2_tmp + alpha_3_tmp
    alpha_0 = alpha_0_tmp/alpha_sum
    alpha_1 = alpha_1_tmp/alpha_sum
    alpha_2 = alpha_2_tmp/alpha_sum
    alpha_3 = alpha_3_tmp/alpha_sum
    
    alpha.append([alpha_0,alpha_1,alpha_2,alpha_3])
    
    print("-----time = %d, alpha %d  ------------------------"  %(k+1, k+1))
    print(alpha_0)
    print(alpha_1)
    print(alpha_2)
    print(alpha_3)


beta = []

beta_0 = 1
beta_1 = 0
beta_2 = 0
beta_3 = 0
print("-----time = 10, beta 10  ------------------------" )
print(beta_0)
print(beta_1)
print(beta_2)
print(beta_3)
    
beta.append([beta_0,beta_1,beta_2,beta_3])

for k in range(9,0,-1):

    ra = r[2*k+0]
    rb = r[2*k+1]

    lambda_pie_00 = np.exp(-1/(2*SQUARE_sigma)*((ra+1)**2 + (rb+1)**2))   # v=00,  -1, -1
    lambda_pie_02 = np.exp(-1/(2*SQUARE_sigma)*((ra-1)**2 + (rb-1)**2))   # v=11,  +1, +1

    lambda_pie_10 = np.exp(-1/(2*SQUARE_sigma)*((ra-1)**2 + (rb+1)**2))   # v=10,  +1, -1
    lambda_pie_12 = np.exp(-1/(2*SQUARE_sigma)*((ra+1)**2 + (rb-1)**2))   # v=01,  -1, +1

    lambda_pie_21 = np.exp(-1/(2*SQUARE_sigma)*((ra+1)**2 + (rb+1)**2))   # v=00,  -1, -1
    lambda_pie_23 = np.exp(-1/(2*SQUARE_sigma)*((ra-1)**2 + (rb-1)**2))   # v=11,  +1, +1

    lambda_pie_31 = np.exp(-1/(2*SQUARE_sigma)*((ra-1)**2 + (rb+1)**2))   # v=10,  +1, -1
    lambda_pie_33 = np.exp(-1/(2*SQUARE_sigma)*((ra+1)**2 + (rb-1)**2))   # v=01,  -1, +1

    beta_0_tmp = lambda_pie_00 * beta_0 + lambda_pie_02 * beta_2
    beta_1_tmp = lambda_pie_10 * beta_0 + lambda_pie_12 * beta_2
    beta_2_tmp = lambda_pie_21 * beta_1 + lambda_pie_23 * beta_3
    beta_3_tmp = lambda_pie_31 * beta_1 + lambda_pie_33 * beta_3


    beta_sum = beta_0_tmp + beta_1_tmp + beta_2_tmp + beta_3_tmp
    beta_0 = beta_0_tmp/beta_sum
    beta_1 = beta_1_tmp/beta_sum
    beta_2 = beta_2_tmp/beta_sum
    beta_3 = beta_3_tmp/beta_sum

    beta.append([beta_0,beta_1,beta_2,beta_3])

    print("-----time = %d, beta %d  ------------------------"  %(k-1, k))
    print(beta_0)
    print(beta_1)
    print(beta_2)
    print(beta_3)


pstrProb =[]

for k in range(0,10):

    ra = r[2*k+0]
    rb = r[2*k+1]

    lambda_pie_00 = np.exp(-1/(2*SQUARE_sigma)*((ra+1)**2 + (rb+1)**2))   # v=00,  -1, -1
    lambda_pie_02 = np.exp(-1/(2*SQUARE_sigma)*((ra-1)**2 + (rb-1)**2))   # v=11,  +1, +1

    lambda_pie_10 = np.exp(-1/(2*SQUARE_sigma)*((ra-1)**2 + (rb+1)**2))   # v=10,  +1, -1
    lambda_pie_12 = np.exp(-1/(2*SQUARE_sigma)*((ra+1)**2 + (rb-1)**2))   # v=01,  -1, +1

    lambda_pie_21 = np.exp(-1/(2*SQUARE_sigma)*((ra+1)**2 + (rb+1)**2))   # v=00,  -1, -1
    lambda_pie_23 = np.exp(-1/(2*SQUARE_sigma)*((ra-1)**2 + (rb-1)**2))   # v=11,  +1, +1

    lambda_pie_31 = np.exp(-1/(2*SQUARE_sigma)*((ra-1)**2 + (rb+1)**2))   # v=10,  +1, -1
    lambda_pie_33 = np.exp(-1/(2*SQUARE_sigma)*((ra+1)**2 + (rb-1)**2))   # v=01,  -1, +1

    
    p0_temp = alpha[k][0] * lambda_pie_00 * beta[9-k][0] + \
    alpha[k][1] * lambda_pie_12 * beta[9-k][2] + \
    alpha[k][2] * lambda_pie_21 * beta[9-k][1] + \
    alpha[k][3] * lambda_pie_33 * beta[9-k][3] 
    
    p1_temp = alpha[k][0] * lambda_pie_02 * beta[9-k][2] + \
    alpha[k][1] * lambda_pie_10 * beta[9-k][0] + \
    alpha[k][2] * lambda_pie_23 * beta[9-k][3] + \
    alpha[k][3] * lambda_pie_31 * beta[9-k][1] 
    
    p0 = p0_temp /(p0_temp + p1_temp)
    p1 = p1_temp /(p0_temp + p1_temp)
    
    pstrProb.append([p0,p1])
    
    print("--- bit %d = %d ---" %(k, p1>p0))
    print(p0)
    print(p1)


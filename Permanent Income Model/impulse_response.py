#%% -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 14:21:55 2018

@author: dongchenzou
"""

import numpy as np
import os
import matplotlib.pyplot as plt

path_1 = 'C:/Users/dongchenzou/Dropbox/'
path_2 = 'Hansen/Robustness/Robust Social Planner/Impulse Response/'
path_3 = 'graphs'
path = path_1 + path_2 + path_3
os.chdir(path)

T = 2000        # Time horizon
S = 2           # Impulse date
σ1 = 0.108*1.33
σ2 = 0.155*1.33

# =============================================================================
#  2.3.4: Impulse response -- Y
# =============================================================================
def time_path():
    "Time path of log income given shock sequence"
    w = np.zeros(T+S)
    X1 = np.zeros(T+S)
    X2 = np.zeros(T+S)
    Y1 = np.zeros(T+S)
    Y2 = np.zeros(T+S)    
    w[S] = 1
    
    for t in range(1, T+S-1):
        X1[t+1] = 0.704 * X1[t] + σ1 * w[t+1]
        Y1[t+1] = Y1[t] + X1[t+1]
        X2[t+1] = X2[t] - 0.154 * X2[t-1] + σ2 * w[t+1]
        Y2[t+1] = X2[t+1]
    return Y1, Y2

Y1, Y2 = time_path()



fig, axes = plt.subplots(2, 1,figsize=(8,8))
plt.subplots_adjust(hspace=0.5)
p_args = {'lw': 2, 'alpha': 0.7}

L = 0.6

for ax in axes:
    ax.grid(alpha=0.5)
    ax.set_xlim(0,40)
    ax.set_xlabel(r'Quarters')
    ax.set_ylim(0, L)

ax = axes[0]
ax.plot(list(range(T)), Y1[S:], 'g-', **p_args)

ax = axes[1]
ax.plot(list(range(T)), Y2[S:], 'b-', **p_args)

fig.savefig('impulse response_Y.eps', format='eps', dpi=1200)

# =============================================================================
# 2.3.5： Impulse response -- C + Y
# =============================================================================
ρ = 0.00663
ν = 0.00373
λ = np.exp(ν - ρ)

D = np.array([[0.704],[0],[-0.154]])
A = np.array([[0.704,0,0],[0,1,-0.154],[0,1,0]])
M = λ * np.matmul(D.T, np.linalg.inv(np.identity(A.shape[0]) - λ*A))
M = M.flatten()

def time_path_2():
    "Time path of log income given shock sequence"
    w = np.zeros(T+S)
    X1 = np.zeros(T+S)
    X2 = np.zeros(T+S)
    C1 = np.zeros(T+S)
    C2 = np.zeros(T+S)
    w[S] = 1
    
    for t in range(1, T+S-1):
        X1[t+1] = 0.704 * X1[t] + σ1 * w[t+1]
        X2[t+1] = X2[t] - 0.154 * X2[t-1] + σ2 * w[t+1]
        #Here I solved the explicit equations of C on X by replacing
        #K terms in Equation (5) with C using Equation (6)
        C1[t+1] = C1[t] - np.exp(ρ - ν) * M[0] * X1[t] + M[0] * X1[t+1]
        C2[t+1] = C2[t] - np.exp(ρ - ν) * (M[1] * X2[t] + M[2] * X2[t-1]) + (M[1] * X2[t+1] + M[2] * X2[t])
       
    return C1, C2

C1, C2 = time_path_2()
C1Y1 = C1 + Y1
C2Y2 = C2 + Y2


fig, axes = plt.subplots(2, 1,figsize=(8,8))
plt.subplots_adjust(hspace=0.5)
p_args = {'lw': 2, 'alpha': 0.7}

L = 0.6

for ax in axes:
    ax.grid(alpha=0.5)
    ax.set_xlim(0,40)
    ax.set_xlabel(r'Quarters')
    ax.set_ylim(0, L)

ax = axes[0]
ax.plot(list(range(T)), C1Y1[S:], 'g-', **p_args)

ax = axes[1]
ax.plot(list(range(T)), C2Y2[S:], 'b-', **p_args)

fig.savefig('impulse response_CY.eps', format='eps', dpi=1200)


print('The number for the first shock is: ',C1Y1[-1])
print('The number for the second shock is: ',C2Y2[-1])

#==============================================================================
# Impulse responses -- C 
#==============================================================================
fig, axes = plt.subplots(2, 1,figsize=(8,8))
plt.subplots_adjust(hspace=0.5)
p_args = {'lw': 2, 'alpha': 0.7}

L = 0.4

for ax in axes:
    ax.grid(alpha=0.5)
    ax.set_xlim(0,40)
    ax.set_xlabel(r'Quarters')
    ax.set_ylim(-L, L)


ax = axes[0]
ax.plot(list(range(T)), C1[S:], 'g-', **p_args)

ax = axes[1]
ax.plot(list(range(T)), C2[S:], 'b-', **p_args)

#==============================================================================
# Discounted cumulative sum for impulse response of C
#==============================================================================
n = np.arange(T)
lam = λ**n
cumsum_C1 = np.cumsum(C1[S:]*lam)
cumsum_C2 = np.cumsum(C2[S:]*lam)

fig, axes = plt.subplots(2, 1,figsize=(8,8))
plt.subplots_adjust(hspace=0.5)
p_args = {'lw': 2, 'alpha': 0.7}


for ax in axes:
    ax.grid(alpha=0.5)
    ax.set_xlim(0,T)
    ax.plot((0,T),(0,0), 'k--')
    ax.set_xlabel(r'Quarters')
    
    
ax = axes[0]
ax.plot(list(range(T)), cumsum_C1, 'g-', **p_args)

ax = axes[1]
ax.plot(list(range(T)), cumsum_C2, 'b-', **p_args)  

fig.savefig('discounted cumulative sum_C.eps', format='eps', dpi=1200)  

#==============================================================================
# Observational Equivalence
#==============================================================================
FcFy_sq = (C1Y1[-1]*0.01)**2 + (C2Y2[-1]*0.01)**2
s = np.arange(0,1000,0.1)
δ = ρ - ν - FcFy_sq*s
max_s = s[np.where(δ > 0)[0][-1]]
print(r'The maximum 1/xi: ', max_s)

fig, ax = plt.subplots(figsize=(10,8))
plt.subplots_adjust(hspace=0.5)
p_args = {'lw': 2, 'alpha': 0.7}
ax.set_xlim(0,max_s)
ax.set_ylim(0,δ[0])
ax.set_xlabel(r'$\frac{1}{\xi}$', size='xx-large')
ax.set_ylabel(r'$\delta$', size='xx-large')

ax.plot(s, δ)

fig.savefig('discount rate on robustness.eps', format='eps', dpi=1200)
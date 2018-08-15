#%% -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 12:31:25 2018

@author: dongchenzou
"""
import numpy as np
import os
import matplotlib.pyplot as plt
from sympy.solvers import solve
from sympy import Symbol, exp, log, symbols, linear_eq_to_matrix


#Set print options for numpy
np.set_printoptions(suppress=True, threshold=3000)


#Define path
path_1 = 'C:/Users/dongchenzou/Dropbox/'
path_2 = 'Hansen/Robustness/Robust Social Planner/Habit Persistence/'
path_3 = 'graphs'
path = path_1 + path_2 + path_3
os.chdir(path)

#Define parameters
T = 2000        # Time horizon
S = 2           # Impulse date
σ1 = 0.108*1.33
σ2 = 0.155*1.33
c = 0
ρ = 0.00663
ν = 0.00373
δ = ρ - ν

#Define shocks
Z0_1 = np.zeros((5,1))
Z0_1[2,0] = σ1
Z0_2 = np.zeros((5,1))
Z0_2[3,0] = σ2

#Define plot attributes
p_args = {'lw': 2, 'alpha': 0.7}



#%%============================================================================

#==============================================================================
# Function: Solve for J matrix and matrix for stable dynamics
#==============================================================================
def solve_habit_persistence(alpha=0.5, eta=2, psi=0.3):
    """
    This function solves the matrix J and stable dynamic matrix A
    in Habit Persistence Section of the RA notes. Here we assume
    I is a 7x7 identity matrx.
    
    Output
    ==========
    J: 7x7 matrix
    A: 5X5 stable dynamic matrix
    N1, N2: stable dynamics for costates
    
    """
    ##== Parameters and Steady State Values ==##
    # Parameters 
    η = eta;
    ψ = psi;
    α = alpha;
    
    # h
    h = Symbol('h')
    Eq = exp(h)*exp(ν) - (exp(-ψ)*exp(h) + (1 - exp(-ψ))*exp(c))
    h = solve(Eq,h)[0]
    print('h =', h)
    
    # u
    u = 1/(1 - η)*log((1 - α)*exp((1 - η)*c) + α*exp((1 - η)*h))
    print('u =', u)
    
    # mh
    mh = Symbol('mh')
    Eq = exp(-δ - ψ - ν)*exp(mh) - (exp(mh) - α*exp((η - 1)*u - η*h))
    mh = solve(Eq,mh)[0]
    print('mh =', mh)
    
    # mk
    mk = Symbol('mk')
    Eq = (1 - α)*exp((η - 1)*u - η*c) - \
         (exp(-δ - ν)*exp(mk) - exp(-δ - ν)*(1 - exp(-ψ))*exp(mh))
    mk = solve(Eq,mk)[0]
    print('mk =', mk)
    
    print('\n')  
    
    
    ##== Construct Ut and Ct ==##
    MKt, MHt, Kt, Ht, X1t, X2t, X2tL1 = symbols('MKt MHt Kt Ht X1t X2t X2tL1')
    MKt1, MHt1, Kt1, Ht1, X1t1, X2t1 = symbols('MKt1 MHt1 Kt1 Ht1 X1t1 X2t1')
    Ct, Ut = symbols('Ct Ut')
    
    # Equation (9)
    Eq1 = Ut - ((1 - α)*exp((η - 1)*(u - c))*Ct + α*exp((η - 1)*(u - h))*Ht)
    
    # Equation (11)
    Eq2 = (1 - α)*exp((η - 1)*u - η*h)*((η - 1)*Ut - η*Ct) - \
          (exp(-δ - ν)*(exp(mk)*MKt1 - (1 - exp(-ψ))*exp(mh)*MHt1) + \
           exp(-δ - ν)*(exp(mk) - (1 - exp(-ψ))*exp(mh))*(0.704*X1t1 - 0.154*X2t))
        
    #Solve the system    
    sol = solve([Eq1, Eq2], Ct, Ut, set=True)[1]
    sol = list(sol)[0]
    Ct, Ut = sol
    
    print('Ct =', Ct)
    print('Ut =', Ut)
    
    print('\n')
    
    
    ##== Solve for Linear system L and J  ==##
    # Equation (12)
    Eq1 = MKt1 - (exp(δ - ρ + ν)*MKt + (0.704*X1t - 0.154*X2tL1))
    
    # Equation (10)
    Eq2 = MHt1 - \
          (exp(δ + ψ + ν - mh)*(exp(mh)*MHt - \
           α*exp((η - 1)*u - η*h)*((η - 1)*Ut - η*Ht)) + \
           (0.704*X1t - 0.154*X2tL1))
    
    # Equation (4)
    Eq3 = Kt1 - (exp(ρ - ν)*Kt - exp(-ν)*Ct)
    
    # Equation (8)
    Eq4 = Ht1 - (exp(-ψ - ν)*Ht + (1 - exp(-ψ - ν))*Ct - (0.704*X1t - 0.154*X2tL1))
    
    # Equation for X
    Eq5 = X1t1 - 0.704*X1t
    Eq6 = X2t1 - (X2t - 0.154*X2tL1)
    
    #Solve the system 
    sol = solve([Eq1, Eq2, Eq3, Eq4, Eq5, Eq6], 
                Ht1, Kt1, MHt1, MKt1, X1t1, X2t1, 
                set=True)
    sol = list(sol[1])[0]
    Ht1, Kt1, MHt1, MKt1, X1t1, X2t1 = sol
    
    #Solve for J
    J,_ = linear_eq_to_matrix([MKt1,MHt1,Kt1,Ht1,X1t1,X2t1,X2t], 
                              MKt, MHt, Kt, Ht, X1t, X2t, X2tL1)
    J = np.asarray(J).astype(float)
    
    
    ##== Get Eigenvalues and Eigenvectors of J ==##
    eigenValues, eigenVectors = np.linalg.eig(J)
    idx = np.abs(eigenValues).argsort()[::-1]   
    eigenValues = eigenValues[idx]
    eigenVectors = eigenVectors[:,idx]
    
    print('The eigenvalues of J are:')
    [print(np.round(x,6)) for x in eigenValues]
    
    print('\n')
    
    
    ##== Initial values and matrix for stable dynamics ==##
    #Get the matrix for stable dynamics by solving the linear combination
    M = eigenVectors[:,np.abs(eigenValues)<=1+10e-10] #Numeric tolerance
    test = M[2:,:]
    sol = np.linalg.solve(test,np.identity(5))
    K = np.matmul(M,sol)
    
    #Get N1 and N2
    N1 = K[:2,:2]
    N2 = K[:2,2:]
    
    
    ##== Check the construction result and get stable dynamic matrix ==##
    print('==== Checking results ====')
    JK = np.matmul(J,K)
    
    #First check
    check_mat = np.hstack([np.identity(2),-N1,-N2])
    check_res = np.matmul(check_mat, JK)
    print(np.round(check_res, 10)==0)
    
    #Stable dynamic matrix
    M_L = np.hstack([np.zeros((5,2)),np.identity(5)])
    A = np.matmul(M_L,JK)
    
    #Second Check
    check_eig = np.linalg.eig(A)[0]
    idx = np.abs(check_eig).argsort()[::-1]
    check_eig = check_eig[idx]
    for n,x in enumerate(check_eig):
        print(np.round(x, 10)==np.round(eigenValues[n+2], 10)) 

    return J, A, N1, N2, Ct, Ut



#==============================================================================
# Functions: Output time path for C, Z and Y
#==============================================================================
def C_and_Z_path(A, N1, N2, Z0, Ct, MKMH_option=False):
    """
    This function outputs the time path of C and Z responses given the 
    intial shock vector Z0.
    
    Output
    ==========
    C_path: the resulting time path of consumption ratio response 
            given Z0.
    Z_path: the resulting time path of shock response given Z0.
    
    """
    Z_path = np.zeros_like(Z0)
    Z_path = np.hstack([Z_path, Z0])
    
    for t in range(T):
        Z = np.matmul(A,Z0)
        Z_path = np.hstack([Z_path,Z])
        Z0 = Z
    
    Z_path = Z_path[:,1:] 
    C_path = np.zeros(T)
    KH_path = Z_path[:2,:]   
    X_path = Z_path[2:,:] 
    MKMH_path = np.matmul(N1,KH_path) + np.matmul(N2,X_path)  
    
    for t in range(T-1):
        MK1 = MKMH_path[0,t+1]
        MH1 = MKMH_path[1,t+1]
        H = KH_path[1,t]
        X11 = X_path[0,t+1]
        X2 = X_path[2,t+1]
        MKt1, MHt1, Ht, X1t1, X2t = symbols('MKt1 MHt1 Ht X1t1 X2t')
        """
        Ct is the explicit formula of consumption ratio process 
        imported from function: solve_habit_persistence
        """
        C = Ct.subs([(MKt1,MK1),
                     (MHt1,MH1),
                     (Ht,H),
                     (X1t1,X11),
                     (X2t,X2)])
        C_path[t] = C
    
    if MKMH_option:    
        return C_path, Z_path, MKMH_path
    else:
        return C_path, Z_path


def Y_path(Z_path):
    """
    Time path of log income given shock sequence Z_path
    
    Output
    ==========
    Y1_path: first shock (permanent)
    Y2_path: second shock (transient)
    
    """
    X1_path = Z_path[2,:-1]
    X2_path = Z_path[3,:-1]
    Y1_path = np.cumsum(X1_path)
    Y2_path = X2_path
    
    return Y1_path, Y2_path



#==============================================================================
# Functions: Setup the figures
#==============================================================================
def create_fig(R, C, fs=(8,8), X=40):
    """
    Create the figure for response plots
    
    Input
    ==========
    R: Number of rows for the subplot space
    C: Number of columns for the subplot space
    fs: figure size
    
    Output
    ==========
    fig, axes: the formatted figure and axes
        
    """
    fig, axes = plt.subplots(R, C, figsize=fs)
    plt.subplots_adjust(hspace=0.5)
    
    for ax in axes:
        ax.grid(alpha=0.5)
        ax.set_xlim(0,X)
        ax.set_xlabel(r'Quarters')
    
    return fig, axes        




#%%============================================================================

#==============================================================================
# Impulse response plots for alpha = 0.5, eta = 0.5, 0.99, 1.5, 2
#==============================================================================
##== Get impulse responses for C and C + Y ==##
#List to append C and C + Y
C1_list = []
C2_list = []
C1Y1_list = []
C2Y2_list = []

eta_list = [0.5, 0.99, 1.5, 2]

for e in eta_list:
    
    J, A, N1, N2, Ct,_ = solve_habit_persistence(eta=e)
    
    ## Impulse response for C    
    C1, Z1 = C_and_Z_path(A, N1, N2, Z0_1, Ct)
    C2, Z2 = C_and_Z_path(A, N1, N2, Z0_2, Ct)
    
    C1_list.append(C1)
    C2_list.append(C2)
    
    
    ## Impulse response for the consumption process C + Y
    Y1,_ = Y_path(Z1)
    _,Y2 = Y_path(Z2)
    
    C1Y1_list.append(C1 + Y1)
    C2Y2_list.append(C2 + Y2)    
    

##== Plot the graphs for the impulse reponses ==##
## Impulse Response plot for C
fig, axes = create_fig(2,1)

ax = axes[0]
ax.set_ylim(-0.1, 0.4)
for n, C1 in enumerate(C1_list):
    ax.plot(list(range(T)), C1, **p_args, 
            label=r'$\eta$ = {}'.format(eta_list[n]))
    ax.legend()

ax = axes[1]
ax.set_ylim(-0.3, 0.1)
for n, C2 in enumerate(C2_list):
    ax.plot(list(range(T)), C2, **p_args, 
            label=r'$\eta$ = {}'.format(eta_list[n]))
    ax.legend()

fig.savefig('habit_persistence_impulse response_C.eps', 
            format='eps', 
            dpi=1200)


## Impulse response plot for C + Y
fig, axes = create_fig(2,1)

ax = axes[0]
ax.set_ylim(0.3, 0.6)
for n, C1Y1 in enumerate(C1Y1_list):
    ax.plot(list(range(T)), C1Y1, **p_args, 
            label=r'$\eta$ = {}'.format(eta_list[n]))
    ax.legend()

ax = axes[1]
ax.set_ylim(-0.1, 0.1)
for n, C2Y2 in enumerate(C2Y2_list):
    ax.plot(list(range(T)), C2Y2, **p_args, 
            label=r'$\eta$ = {}'.format(eta_list[n]))
    ax.legend()

fig.savefig('habit_persistence_impulse response_CY.eps', 
            format='eps', 
            dpi=1200)

"""
Note
=======
For replication purpose, the eigenvalues for eta = 2 are:
1.39605, 1.0029, 1., 0.809839, 0.720546, 0.704, 0.190161

"""


#==============================================================================
# Impulse response plots for alpha = 0.9, eta = 0.5, 0.99, 1.5, 2
#==============================================================================
##== Get impulse responses for C and C + Y ==##
#List to append C and C + Y
C1_list = []
C2_list = []
C1Y1_list = []
C2Y2_list = []

eta_list = [0.5, 0.99, 1.5, 2]

for e in eta_list:
    
    J, A, N1, N2, Ct,_ = solve_habit_persistence(alpha=0.9, eta=e)
    
    ## Impulse response for C    
    C1, Z1 = C_and_Z_path(A, N1, N2, Z0_1, Ct)
    C2, Z2 = C_and_Z_path(A, N1, N2, Z0_2, Ct)
    
    C1_list.append(C1)
    C2_list.append(C2)
    
    
    ## Impulse response for the consumption process C + Y
    Y1,_ = Y_path(Z1)
    _,Y2 = Y_path(Z2)
    
    C1Y1_list.append(C1 + Y1)
    C2Y2_list.append(C2 + Y2)    
    

##== Plot the graphs for the impulse reponses ==##
## Impulse Response plot for C
fig, axes = create_fig(2,1)

ax = axes[0]
ax.set_ylim(-0.1, 0.9)
for n, C1 in enumerate(C1_list):
    ax.plot(list(range(T)), C1, **p_args, 
            label=r'$\eta$ = {}'.format(eta_list[n]))
    ax.legend()

ax = axes[1]
ax.set_ylim(-0.6, 0.1)
for n, C2 in enumerate(C2_list):
    ax.plot(list(range(T)), C2, **p_args, 
            label=r'$\eta$ = {}'.format(eta_list[n]))
    ax.legend()

fig.savefig('habit_persistence_impulse response_C_alpha=0.9.eps', 
            format='eps', 
            dpi=1200)


## Impulse response plot for C + Y
fig, axes = create_fig(2,1)

ax = axes[0]
ax.set_ylim(0.4, 1.1)
for n, C1Y1 in enumerate(C1Y1_list):
    ax.plot(list(range(T)), C1Y1, **p_args, 
            label=r'$\eta$ = {}'.format(eta_list[n]))
    ax.legend()

ax = axes[1]
ax.set_ylim(-0.4, 0.1)
for n, C2Y2 in enumerate(C2Y2_list):
    ax.plot(list(range(T)), C2Y2, **p_args, 
            label=r'$\eta$ = {}'.format(eta_list[n]))
    ax.legend()

fig.savefig('habit_persistence_impulse response_CY_alpha=0.9.eps', 
            format='eps', 
            dpi=1200)



#==============================================================================
# Impulse response plots for alpha = 0.5, eta = 10, 20
#==============================================================================
##== Get impulse responses for C and C + Y ==##
#List to append C and C + Y
C1_list = []
C2_list = []
C1Y1_list = []
C2Y2_list = []

eta_list = [10, 20]

for e in eta_list:
    
    J, A, N1, N2, Ct,_ = solve_habit_persistence(alpha=0.5, eta=e)
    
    ## Impulse response for C    
    C1, Z1 = C_and_Z_path(A, N1, N2, Z0_1, Ct)
    C2, Z2 = C_and_Z_path(A, N1, N2, Z0_2, Ct)
    
    C1_list.append(C1)
    C2_list.append(C2)
    
    
    ## Impulse response for the consumption process C + Y    
    Y1,_ = Y_path(Z1)
    _,Y2 = Y_path(Z2)
    
    C1Y1_list.append(C1 + Y1)
    C2Y2_list.append(C2 + Y2)    
    

##== Plot the graphs for the impulse reponses ==##
## Impulse Response plot for C
fig, axes = create_fig(2,1)

ax = axes[0]
ax.set_ylim(-0.2, 0.3)
for n, C1 in enumerate(C1_list):
    ax.plot(list(range(T)), C1, **p_args, 
            label=r'$\eta$ = {}'.format(eta_list[n]))
    ax.legend()

ax = axes[1]
ax.set_ylim(-0.2, 0.3)
for n, C2 in enumerate(C2_list):
    ax.plot(list(range(T)), C2, **p_args, 
            label=r'$\eta$ = {}'.format(eta_list[n]))
    ax.legend()

fig.savefig('habit_persistence_impulse response_C_big eta.eps', 
            format='eps', 
            dpi=1200)


## Impulse response plot for C + Y
fig, axes = create_fig(2,1)

ax = axes[0]
ax.set_ylim(0.2, 0.6)
for n, C1Y1 in enumerate(C1Y1_list):
    ax.plot(list(range(T)), C1Y1, **p_args, 
            label=r'$\eta$ = {}'.format(eta_list[n]))
    ax.legend()

ax = axes[1]
ax.set_ylim(-0.1, 0.2)
for n, C2Y2 in enumerate(C2Y2_list):
    ax.plot(list(range(T)), C2Y2, **p_args, 
            label=r'$\eta$ = {}'.format(eta_list[n]))
    ax.legend()

fig.savefig('habit_persistence_impulse response_CY_big eta.eps', 
            format='eps', 
            dpi=1200)



#==============================================================================
# Impulse response plots for alpha = 0.5, eta = 2, 4, 8, 16.
#==============================================================================
##== Get impulse responses for C and C + Y ==##
#List to append C and C + Y
C1_list = []
C2_list = []
C1Y1_list = []
C2Y2_list = []

eta_list = [2, 4, 8, 16]

for e in eta_list:
    
    J, A, N1, N2, Ct,_ = solve_habit_persistence(alpha=0.5, eta=e)
    
    ## Impulse response for C
    C1, Z1 = C_and_Z_path(A, N1, N2, Z0_1, Ct)
    C2, Z2 = C_and_Z_path(A, N1, N2, Z0_2, Ct)
    
    C1_list.append(C1)
    C2_list.append(C2)
    
    
    ## Impulse response for the consumption process C + Y    
    Y1,_ = Y_path(Z1)
    _,Y2 = Y_path(Z2)
    
    C1Y1_list.append(C1 + Y1)
    C2Y2_list.append(C2 + Y2)    
    

##== Plot the graphs for the impulse reponses ==##
## Impulse Response plot for C
fig, axes = create_fig(2,1)

ax = axes[0]
ax.set_ylim(-0.1, 0.4)
for n, C1 in enumerate(C1_list):
    ax.plot(list(range(T)), C1, **p_args, 
            label=r'$\eta$ = {}'.format(eta_list[n]))
    ax.legend()

ax = axes[1]
ax.set_ylim(-0.2, 0.1)
for n, C2 in enumerate(C2_list):
    ax.plot(list(range(T)), C2, **p_args, 
            label=r'$\eta$ = {}'.format(eta_list[n]))
    ax.legend()

fig.savefig('habit_persistence_impulse response_C_eta=2-4-8-16.eps', 
            format='eps', 
            dpi=1200)


## Impulse response plot for C + Y
fig, axes = create_fig(2,1)

ax = axes[0]
ax.set_ylim(0.2, 0.5)
for n, C1Y1 in enumerate(C1Y1_list):
    ax.plot(list(range(T)), C1Y1, **p_args, 
            label=r'$\eta$ = {}'.format(eta_list[n]))
    ax.legend()

ax = axes[1]
ax.set_ylim(-0.1, 0.2)
for n, C2Y2 in enumerate(C2Y2_list):
    ax.plot(list(range(T)), C2Y2, **p_args, 
            label=r'$\eta$ = {}'.format(eta_list[n]))
    ax.legend()

fig.savefig('habit_persistence_impulse response_CY_eta=2-4-8-16.eps', 
            format='eps', 
            dpi=1200)



#==============================================================================
# psi = 100, alpha = 0.99, 0.5, 0.2, 0.1, eta = 0.99
#==============================================================================
##== Get impulse responses for C and C + Y ==##
#List to append C and C + Y
C1_list = []
C2_list = []
C1Y1_list = []
C2Y2_list = []

alpha_list = [0.99, 0.5, 0.2, 0.1]

for a in alpha_list:
    
    J, A, N1, N2, Ct,_ = solve_habit_persistence(alpha=a, eta=0.99, psi=100)
    
    ## Impulse response for C
    C1, Z1 = C_and_Z_path(A, N1, N2, Z0_1, Ct)
    C2, Z2 = C_and_Z_path(A, N1, N2, Z0_2, Ct)
    
    C1_list.append(C1)
    C2_list.append(C2)
    
    
    ## Impulse response for the consumption process C + Y    
    Y1,_ = Y_path(Z1)
    _,Y2 = Y_path(Z2)
    
    C1Y1_list.append(C1 + Y1)
    C2Y2_list.append(C2 + Y2)    
    

##== Plot the graphs for the impulse reponses ==##
## Impulse Response plot for C
fig, axes = create_fig(2,1)

ax = axes[0]
ax.set_ylim(-0.1, 0.4)
for n, C1 in enumerate(C1_list):
    ax.plot(list(range(T)), C1, **p_args, 
            label=r'$\alpha$ = {}'.format(alpha_list[n]))
    ax.legend()

ax = axes[1]
ax.set_ylim(-0.3, 0.1)
for n, C2 in enumerate(C2_list):
    ax.plot(list(range(T)), C2, **p_args, 
            label=r'$\alpha$ = {}'.format(alpha_list[n]))
    ax.legend()

fig.savefig('habit_persistence_impulse response_C_psi=100_eta=0.99.eps', 
            format='eps', 
            dpi=1200)


## Impulse response plot for C + Y
fig, axes = create_fig(2,1)

ax = axes[0]
ax.set_ylim(0.3, 0.6)
for n, C1Y1 in enumerate(C1Y1_list):
    ax.plot(list(range(T)), C1Y1, **p_args, 
            label=r'$\alpha$ = {}'.format(alpha_list[n]))
    ax.legend()

ax = axes[1]
ax.set_ylim(-0.01, 0.1)
for n, C2Y2 in enumerate(C2Y2_list):
    ax.plot(list(range(T)), C2Y2, **p_args, 
            label=r'$\alpha$ = {}'.format(alpha_list[n]))
    ax.legend()

fig.savefig('habit_persistence_impulse response_CY_psi=100_eta=0.99.eps', 
            format='eps', 
            dpi=1200)




#==============================================================================
# psi = 0.3; alpha = 0.5, 0.9; eta = 10, 20, 40, (add. 60)
#==============================================================================
##== Get impulse responses for C and C + Y ==##
#List to append C and C + Y
C1_list = []
C2_list = []
C1Y1_list = []
C2Y2_list = []

alpha_list = [0.5, 0.9]
eta_list = [10, 20, 40, 60]
elen = len(eta_list)

for a in alpha_list:
    for e in eta_list:
        
        J, A, N1, N2, Ct,_ = solve_habit_persistence(alpha=a, eta=e)
        
        ## Impulse response for C    
        C1, Z1 = C_and_Z_path(A, N1, N2, Z0_1, Ct)
        C2, Z2 = C_and_Z_path(A, N1, N2, Z0_2, Ct)
        
        C1_list.append(C1)
        C2_list.append(C2)
        
        
        ## Impulse response for the consumption process C + Y    
        Y1,_ = Y_path(Z1)
        _,Y2 = Y_path(Z2)
        
        C1Y1_list.append(C1 + Y1)
        C2Y2_list.append(C2 + Y2)    
    


##== Plot the graphs for the impulse reponses ==##

## Impulse Response plot for C
# alpha = 0.5
fig, axes = create_fig(2,1,X=80)

ax = axes[0]
for n, C1 in enumerate(C1_list[:elen-1]):
    ax.plot(list(range(T)), C1, **p_args, 
            label=r'$\alpha$ = {0}, $\eta$ = {1}'.format(alpha_list[0], eta_list[n]))
    ax.legend()

ax = axes[1]
for n, C1 in enumerate(C2_list[:elen-1]):
    ax.plot(list(range(T)), C1, **p_args, 
            label=r'$\alpha$ = {0}, $\eta$ = {1}'.format(alpha_list[0], eta_list[n]))
    ax.legend()

fig.savefig('habit_persistence_impulse response_C_big eta_alpha=0.5.eps', 
            format='eps', 
            dpi=1200)

# alpha = 0.9
fig, axes = create_fig(2,1,X=80)

ax = axes[0]
for n, C1 in enumerate(C1_list[elen:-1]):
    ax.plot(list(range(T)), C1, **p_args, 
            label=r'$\alpha$ = {0}, $\eta$ = {1}'.format(alpha_list[1], eta_list[n]))
    ax.legend()

ax = axes[1]
for n, C1 in enumerate(C2_list[elen:-1]):
    ax.plot(list(range(T)), C1, **p_args, 
            label=r'$\alpha$ = {0}, $\eta$ = {1}'.format(alpha_list[1], eta_list[n]))
    ax.legend()

fig.savefig('habit_persistence_impulse response_C_big eta_alpha=0.9.eps', 
            format='eps', 
            dpi=1200)


## Impulse response plot for C + Y
# alpha = 0.5
fig, axes = create_fig(2,1)

ax = axes[0]
for n, C1Y1 in enumerate(C1Y1_list[:elen-1]):
    ax.plot(list(range(T)), C1Y1, **p_args, 
            label=r'$\alpha$ = {0}, $\eta$ = {1}'.format(alpha_list[0], eta_list[n]))
    ax.legend()

ax = axes[1]
for n, C2Y2 in enumerate(C2Y2_list[:elen-1]):
    ax.plot(list(range(T)), C2Y2, **p_args, 
            label=r'$\alpha$ = {0}, $\eta$ = {1}'.format(alpha_list[0], eta_list[n]))
    ax.legend()

fig.savefig('habit_persistence_impulse response_CY_big eta_alpha=0.5.eps', 
            format='eps', 
            dpi=1200)

# alpha = 0.9
fig, axes = create_fig(2,1)

ax = axes[0]
for n, C1Y1 in enumerate(C1Y1_list[elen:-1]):
    ax.plot(list(range(T)), C1Y1, **p_args, 
            label=r'$\alpha$ = {0}, $\eta$ = {1}'.format(alpha_list[1], eta_list[n]))
    ax.legend()

ax = axes[1]
for n, C2Y2 in enumerate(C2Y2_list[elen:-1]):
    ax.plot(list(range(T)), C2Y2, **p_args, 
            label=r'$\alpha$ = {0}, $\eta$ = {1}'.format(alpha_list[1], eta_list[n]))
    ax.legend()

fig.savefig('habit_persistence_impulse response_CY_big eta_alpha=0.9.eps', 
            format='eps', 
            dpi=1200)


## Impulse response plot for C + Y over T = 80 and add eta = 60
# alpha = 0.5
fig, axes = create_fig(2,1,X=80)

ax = axes[0]
for n, C1Y1 in enumerate(C1Y1_list[:elen]):
    ax.plot(list(range(T)), C1Y1, **p_args, 
            label=r'$\alpha$ = {0}, $\eta$ = {1}'.format(alpha_list[0], eta_list[n]))
    ax.legend()

ax = axes[1]
for n, C2Y2 in enumerate(C2Y2_list[:elen]):
    ax.plot(list(range(T)), C2Y2, **p_args, 
            label=r'$\alpha$ = {0}, $\eta$ = {1}'.format(alpha_list[0], eta_list[n]))
    ax.legend()

fig.savefig('habit_persistence_impulse response_CY_big eta_alpha=0.5_long horizon.eps', 
            format='eps', 
            dpi=1200)

# alpha = 0.9
fig, axes = create_fig(2,1,X=80)

ax = axes[0]
ax.set_ylim(-0.01, 0.51)
for n, C1Y1 in enumerate(C1Y1_list[elen:]):
    ax.plot(list(range(T)), C1Y1, **p_args, 
            label=r'$\alpha$ = {0}, $\eta$ = {1}'.format(alpha_list[1], eta_list[n]))
    ax.legend()

ax = axes[1]
ax.set_ylim(-0.01, 0.51)
for n, C2Y2 in enumerate(C2Y2_list[elen:]):
    ax.plot(list(range(T)), C2Y2, **p_args, 
            label=r'$\alpha$ = {0}, $\eta$ = {1}'.format(alpha_list[1], eta_list[n]))
    ax.legend()

fig.savefig('habit_persistence_impulse response_CY_big eta_alpha=0.9_long horizon.eps', 
            format='eps', 
            dpi=1200)




#==============================================================================
# psi = 0.3; alpha = 0.1; eta = 10, 20, 40, 60
#==============================================================================
C1_list = []
C2_list = []
C1Y1_list = []
C2Y2_list = []

eta_list = [10, 20, 40, 60]



for e in eta_list:
        
    J, A, N1, N2, Ct,_ = solve_habit_persistence(alpha=0.1, eta=e)
    
    ## Impulse response for C    
    C1, Z1 = C_and_Z_path(A, N1, N2, Z0_1, Ct)
    C2, Z2 = C_and_Z_path(A, N1, N2, Z0_2, Ct)
    
    C1_list.append(C1)
    C2_list.append(C2)
    
    
    ## Impulse response for the consumption process C + Y    
    Y1,_ = Y_path(Z1)
    _,Y2 = Y_path(Z2)
    
    C1Y1_list.append(C1 + Y1)
    C2Y2_list.append(C2 + Y2)  
    
    
    
##== Plot the graphs for the impulse reponses ==##

## Impulse Response plot for C
fig, axes = create_fig(2,1,X=80)

ax = axes[0]
for n, C1 in enumerate(C1_list):
    ax.plot(list(range(T)), C1, **p_args, 
            label=r'$\alpha$ = {0}, $\eta$ = {1}'.format(0.1, eta_list[n]))
    ax.legend()

ax = axes[1]
for n, C1 in enumerate(C2_list):
    ax.plot(list(range(T)), C1, **p_args, 
            label=r'$\alpha$ = {0}, $\eta$ = {1}'.format(0.1, eta_list[n]))
    ax.legend()

fig.savefig('habit_persistence_impulse response_C_big eta_alpha=0.1_psi=0.3.eps', 
            format='eps', 
            dpi=1200)   


## Impulse response plot for C + Y
fig, axes = create_fig(2,1)

ax = axes[0]
for n, C1Y1 in enumerate(C1Y1_list):
    ax.plot(list(range(T)), C1Y1, **p_args, 
            label=r'$\alpha$ = {0}, $\eta$ = {1}'.format(0.1, eta_list[n]))
    ax.legend()

ax = axes[1]
for n, C2Y2 in enumerate(C2Y2_list):
    ax.plot(list(range(T)), C2Y2, **p_args, 
            label=r'$\alpha$ = {0}, $\eta$ = {1}'.format(0.1, eta_list[n]))
    ax.legend()

fig.savefig('habit_persistence_impulse response_CY_big eta_alpha=0.1_psi=0.3.eps', 
            format='eps', 
            dpi=1200)


## Impulse response plot for C + Y over T = 80
fig, axes = create_fig(2,1,X=80)

ax = axes[0]
ax.set_ylim(-0.01, 0.51)
for n, C1Y1 in enumerate(C1Y1_list):
    ax.plot(list(range(T)), C1Y1, **p_args, 
            label=r'$\alpha$ = {0}, $\eta$ = {1}'.format(0.1, eta_list[n]))
    ax.legend()

ax = axes[1]
ax.set_ylim(-0.01, 0.51)
for n, C2Y2 in enumerate(C2Y2_list):
    ax.plot(list(range(T)), C2Y2, **p_args, 
            label=r'$\alpha$ = {0}, $\eta$ = {1}'.format(0.1, eta_list[n]))
    ax.legend()

fig.savefig('habit_persistence_impulse response_CY_big eta_alpha=0.1_psi=0.3_long horizon.eps', 
            format='eps', 
            dpi=1200)





#==============================================================================
# psi = 0.3; alpha = 0.1; eta = 10, 25, 50, 100
#==============================================================================
C1_list = []
C2_list = []
C1Y1_list = []
C2Y2_list = []

eta_list = [10, 25, 50, 100]



for e in eta_list:
        
    J, A, N1, N2, Ct,_ = solve_habit_persistence(alpha=0.1, eta=e)
    
    ## Impulse response for C    
    C1, Z1 = C_and_Z_path(A, N1, N2, Z0_1, Ct)
    C2, Z2 = C_and_Z_path(A, N1, N2, Z0_2, Ct)
    
    C1_list.append(C1)
    C2_list.append(C2)
    
    
    ## Impulse response for the consumption process C + Y    
    Y1,_ = Y_path(Z1)
    _,Y2 = Y_path(Z2)
    
    C1Y1_list.append(C1 + Y1)
    C2Y2_list.append(C2 + Y2)  
    


## Impulse response plot for C + Y over T = 80
fig, axes = create_fig(2,1,X=80)

ax = axes[0]
ax.set_ylim(-0.01, 0.51)
for n, C1Y1 in enumerate(C1Y1_list):
    ax.plot(list(range(T)), C1Y1, **p_args, 
            label=r'$\alpha$ = {0}, $\eta$ = {1}'.format(0.1, eta_list[n]))
    ax.legend()

ax = axes[1]
ax.set_ylim(-0.01, 0.51)
for n, C2Y2 in enumerate(C2Y2_list):
    ax.plot(list(range(T)), C2Y2, **p_args, 
            label=r'$\alpha$ = {0}, $\eta$ = {1}'.format(0.1, eta_list[n]))
    ax.legend()

fig.savefig('habit_persistence_impulse response_CY_big eta 2_alpha=0.1_psi=0.3_long horizon.eps', 
            format='eps', 
            dpi=1200)




#==============================================================================
# psi = 0.4; alpha = 0.1; eta = 10, 20, 40, 60
#==============================================================================
C1_list = []
C2_list = []
C1Y1_list = []
C2Y2_list = []

eta_list = [10, 20, 40, 60]



for e in eta_list:
        
    J, A, N1, N2, Ct,_ = solve_habit_persistence(alpha=0.1, psi=0.4, eta=e)
    
    ## Impulse response for C    
    C1, Z1 = C_and_Z_path(A, N1, N2, Z0_1, Ct)
    C2, Z2 = C_and_Z_path(A, N1, N2, Z0_2, Ct)
    
    C1_list.append(C1)
    C2_list.append(C2)
    
    
    ## Impulse response for the consumption process C + Y    
    Y1,_ = Y_path(Z1)
    _,Y2 = Y_path(Z2)
    
    C1Y1_list.append(C1 + Y1)
    C2Y2_list.append(C2 + Y2)  
    


## Impulse response plot for C + Y over T = 80
fig, axes = create_fig(2,1,X=80)

ax = axes[0]
ax.set_ylim(-0.01, 0.51)
for n, C1Y1 in enumerate(C1Y1_list):
    ax.plot(list(range(T)), C1Y1, **p_args, 
            label=r'$\alpha$ = {0}, $\eta$ = {1}'.format(0.1, eta_list[n]))
    ax.legend()

ax = axes[1]
ax.set_ylim(-0.01, 0.51)
for n, C2Y2 in enumerate(C2Y2_list):
    ax.plot(list(range(T)), C2Y2, **p_args, 
            label=r'$\alpha$ = {0}, $\eta$ = {1}'.format(0.1, eta_list[n]))
    ax.legend()

fig.savefig('habit_persistence_impulse response_CY_big eta_alpha=0.1_psi=0.4_long horizon.eps', 
            format='eps', 
            dpi=1200)





#==============================================================================
# psi = 0.4; alpha = 0.1; eta = 10, 25, 50, 100
#==============================================================================
C1_list = []
C2_list = []
C1Y1_list = []
C2Y2_list = []

eta_list = [10, 25, 50, 100]



for e in eta_list:
        
    J, A, N1, N2, Ct,_ = solve_habit_persistence(alpha=0.1, psi=0.4, eta=e)
    
    ## Impulse response for C    
    C1, Z1 = C_and_Z_path(A, N1, N2, Z0_1, Ct)
    C2, Z2 = C_and_Z_path(A, N1, N2, Z0_2, Ct)
    
    C1_list.append(C1)
    C2_list.append(C2)
    
    
    ## Impulse response for the consumption process C + Y    
    Y1,_ = Y_path(Z1)
    _,Y2 = Y_path(Z2)
    
    C1Y1_list.append(C1 + Y1)
    C2Y2_list.append(C2 + Y2)  
    


## Impulse response plot for C + Y over T = 80
fig, axes = create_fig(2,1,X=80)

ax = axes[0]
ax.set_ylim(-0.01, 0.51)
for n, C1Y1 in enumerate(C1Y1_list):
    ax.plot(list(range(T)), C1Y1, **p_args, 
            label=r'$\alpha$ = {0}, $\eta$ = {1}'.format(0.1, eta_list[n]))
    ax.legend()

ax = axes[1]
ax.set_ylim(-0.01, 0.51)
for n, C2Y2 in enumerate(C2Y2_list):
    ax.plot(list(range(T)), C2Y2, **p_args, 
            label=r'$\alpha$ = {0}, $\eta$ = {1}'.format(0.1, eta_list[n]))
    ax.legend()

fig.savefig('habit_persistence_impulse response_CY_big eta 2_alpha=0.1_psi=0.4_long horizon.eps', 
            format='eps', 
            dpi=1200)





#==============================================================================
# psi = 0.3; alpha = .1, .3, .5, .7, .9; eta = 40
#==============================================================================
C1_list = []
C2_list = []
C1Y1_list = []
C2Y2_list = []

alpha_list = [0.1, 0.3, 0.5, 0.7, 0.9]



for a in alpha_list:
        
    J, A, N1, N2, Ct,_ = solve_habit_persistence(alpha=a, psi=0.3, eta=40)
    
    ## Impulse response for C    
    C1, Z1 = C_and_Z_path(A, N1, N2, Z0_1, Ct)
    C2, Z2 = C_and_Z_path(A, N1, N2, Z0_2, Ct)
    
    C1_list.append(C1)
    C2_list.append(C2)
    
    
    ## Impulse response for the consumption process C + Y    
    Y1,_ = Y_path(Z1)
    _,Y2 = Y_path(Z2)
    
    C1Y1_list.append(C1 + Y1)
    C2Y2_list.append(C2 + Y2)


##== Plot the graphs for the impulse reponses ==##

## Impulse Response plot for C
fig, axes = create_fig(2,1,X=80)

ax = axes[0]
for n, C1 in enumerate(C1_list):
    ax.plot(list(range(T)), C1, **p_args, 
            label=r'$\alpha$ = {0}, $\eta$ = {1}'.format(alpha_list[n], 40))
    ax.legend()

ax = axes[1]
for n, C1 in enumerate(C2_list):
    ax.plot(list(range(T)), C1, **p_args, 
            label=r'$\alpha$ = {0}, $\eta$ = {1}'.format(alpha_list[n], 40))
    ax.legend()

fig.savefig('habit_persistence_impulse response_C_eta=40.eps', 
            format='eps', 
            dpi=1200)   


## Impulse response plot for C + Y
fig, axes = create_fig(2,1)

ax = axes[0]
for n, C1Y1 in enumerate(C1Y1_list):
    ax.plot(list(range(T)), C1Y1, **p_args, 
            label=r'$\alpha$ = {0}, $\eta$ = {1}'.format(alpha_list[n], 40))
    ax.legend()

ax = axes[1]
for n, C2Y2 in enumerate(C2Y2_list):
    ax.plot(list(range(T)), C2Y2, **p_args, 
            label=r'$\alpha$ = {0}, $\eta$ = {1}'.format(alpha_list[n], 40))
    ax.legend()

fig.savefig('habit_persistence_impulse response_CY_eta=40.eps', 
            format='eps', 
            dpi=1200)


## Impulse response plot for C + Y over T = 80 
fig, axes = create_fig(2,1,X=80)

ax = axes[0]
ax.set_ylim(-0.01, 0.51)
for n, C1Y1 in enumerate(C1Y1_list):
    ax.plot(list(range(T)), C1Y1, **p_args, 
            label=r'$\alpha$ = {0}, $\eta$ = {1}'.format(alpha_list[n], 40))
    ax.legend()

ax = axes[1]
ax.set_ylim(-0.01, 0.51)
for n, C2Y2 in enumerate(C2Y2_list):
    ax.plot(list(range(T)), C2Y2, **p_args, 
            label=r'$\alpha$ = {0}, $\eta$ = {1}'.format(alpha_list[n], 40))
    ax.legend()

fig.savefig('habit_persistence_impulse response_CY_eta=40_long horizon.eps', 
            format='eps', 
            dpi=1200)



#==============================================================================
# psi = 0.4; alpha = .1, .3, .5, .7, .9; eta = 50
#==============================================================================
C1_list = []
C2_list = []
C1Y1_list = []
C2Y2_list = []

alpha_list = [0.1, 0.3, 0.5, 0.7, 0.9]



for a in alpha_list:
        
    J, A, N1, N2, Ct,_ = solve_habit_persistence(alpha=a, psi=0.4, eta=50)
    
    ## Impulse response for C    
    C1, Z1 = C_and_Z_path(A, N1, N2, Z0_1, Ct)
    C2, Z2 = C_and_Z_path(A, N1, N2, Z0_2, Ct)
    
    C1_list.append(C1)
    C2_list.append(C2)
    
    
    ## Impulse response for the consumption process C + Y    
    Y1,_ = Y_path(Z1)
    _,Y2 = Y_path(Z2)
    
    C1Y1_list.append(C1 + Y1)
    C2Y2_list.append(C2 + Y2)


## Impulse response plot for C + Y over T = 80 
fig, axes = create_fig(2,1,X=80)

ax = axes[0]
ax.set_ylim(-0.01, 0.51)
for n, C1Y1 in enumerate(C1Y1_list):
    ax.plot(list(range(T)), C1Y1, **p_args, 
            label=r'$\alpha$ = {0}, $\eta$ = {1}'.format(alpha_list[n], 50))
    ax.legend()

ax = axes[1]
ax.set_ylim(-0.01, 0.51)
for n, C2Y2 in enumerate(C2Y2_list):
    ax.plot(list(range(T)), C2Y2, **p_args, 
            label=r'$\alpha$ = {0}, $\eta$ = {1}'.format(alpha_list[n], 50))
    ax.legend()

fig.savefig('habit_persistence_impulse response_CY_eta=50_psi=0.4_long horizon.eps', 
            format='eps', 
            dpi=1200)






#==============================================================================
# alpha = .1, eta = 60 and psi = .1, .2, .4, .8.  (plot MK)
#==============================================================================
MK1_list = []
MK2_list = []
C1Y1_list = []
C2Y2_list = []

psi_list = [.1, .2, .4, .8]



for p in psi_list:
        
    J, A, N1, N2, Ct,_ = solve_habit_persistence(alpha=0.1, eta=60, psi=p)
    
    ## Impulse response for MK   
    C1, Z1, MKMH1 = C_and_Z_path(A, N1, N2, Z0_1, Ct, MKMH_option=True)
    C2, Z2, MKMH2 = C_and_Z_path(A, N1, N2, Z0_2, Ct, MKMH_option=True)
    
    MK1_list.append(MKMH1[0,:-1])
    MK2_list.append(MKMH2[0,:-1])
    
    
    ## Impulse response for the consumption process C + Y    
    Y1,_ = Y_path(Z1)
    _,Y2 = Y_path(Z2)
    
    C1Y1_list.append(C1 + Y1)
    C2Y2_list.append(C2 + Y2)  




## Impulse response plot for MK
fig, axes = create_fig(2,1)

ax = axes[0]
for n, MK1 in enumerate(MK1_list):
    ax.plot(list(range(T)), MK1, **p_args, 
            label=r'$\psi$ = {}'.format(psi_list[n]))
    ax.legend()

ax = axes[1]
for n, MK2 in enumerate(MK2_list):
    ax.plot(list(range(T)), MK2, **p_args, 
            label=r'$\psi$ = {}'.format(psi_list[n]))
    ax.legend()

fig.savefig('habit_persistence_impulse response_MK_alpha=0.1_eta=60.eps', 
            format='eps', 
            dpi=1200)


## Impulse response plot for C + Y
fig, axes = create_fig(2,1)

ax = axes[0]
ax.set_ylim(0., 0.5)
for n, C1Y1 in enumerate(C1Y1_list):
    ax.plot(list(range(T)), C1Y1, **p_args, 
            label=r'$\psi$ = {}'.format(psi_list[n]))
    ax.legend()

ax = axes[1]
ax.set_ylim(-0.1, 0.2)
for n, C2Y2 in enumerate(C2Y2_list):
    ax.plot(list(range(T)), C2Y2, **p_args, 
            label=r'$\psi$ = {}'.format(psi_list[n]))
    ax.legend()

fig.savefig('habit_persistence_impulse response_CY_alpha=0.1_eta=60.eps', 
            format='eps', 
            dpi=1200)





#==============================================================================
# alpha = .1, eta = 80 and psi = .1, .2, .4, .8. 
#==============================================================================
C1Y1_list = []
C2Y2_list = []

psi_list = [.1, .2, .4, .8]


for p in psi_list:
        
    J, A, N1, N2, Ct,_ = solve_habit_persistence(alpha=0.1, eta=80, psi=p)
    
    ## Impulse response for MK   
    C1, Z1 = C_and_Z_path(A, N1, N2, Z0_1, Ct)
    C2, Z2 = C_and_Z_path(A, N1, N2, Z0_2, Ct)

    ## Impulse response for the consumption process C + Y    
    Y1,_ = Y_path(Z1)
    _,Y2 = Y_path(Z2)
    
    C1Y1_list.append(C1 + Y1)
    C2Y2_list.append(C2 + Y2)  


## Impulse response plot for C + Y
fig, axes = create_fig(2,1)

ax = axes[0]
ax.set_ylim(0., 0.5)
for n, C1Y1 in enumerate(C1Y1_list):
    ax.plot(list(range(T)), C1Y1, **p_args, 
            label=r'$\psi$ = {}'.format(psi_list[n]))
    ax.legend()

ax = axes[1]
ax.set_ylim(-0.1, 0.21)
for n, C2Y2 in enumerate(C2Y2_list):
    ax.plot(list(range(T)), C2Y2, **p_args, 
            label=r'$\psi$ = {}'.format(psi_list[n]))
    ax.legend()

fig.savefig('habit_persistence_impulse response_CY_alpha=0.1_eta=80.eps', 
            format='eps', 
            dpi=1200)





#==============================================================================
# alpha = .1, eta = 100 and psi = .1, .2, .4, .8. 
#==============================================================================
C1Y1_list = []
C2Y2_list = []

psi_list = [.1, .2, .4, .8]


for p in psi_list:
        
    J, A, N1, N2, Ct,_ = solve_habit_persistence(alpha=0.1, eta=100, psi=p)
    
    ## Impulse response for MK   
    C1, Z1 = C_and_Z_path(A, N1, N2, Z0_1, Ct)
    C2, Z2 = C_and_Z_path(A, N1, N2, Z0_2, Ct)

    ## Impulse response for the consumption process C + Y    
    Y1,_ = Y_path(Z1)
    _,Y2 = Y_path(Z2)
    
    C1Y1_list.append(C1 + Y1)
    C2Y2_list.append(C2 + Y2)  


## Impulse response plot for C + Y
fig, axes = create_fig(2,1)

ax = axes[0]
ax.set_ylim(0., 0.5)
for n, C1Y1 in enumerate(C1Y1_list):
    ax.plot(list(range(T)), C1Y1, **p_args, 
            label=r'$\psi$ = {}'.format(psi_list[n]))
    ax.legend()

ax = axes[1]
ax.set_ylim(-0.1, 0.21)
for n, C2Y2 in enumerate(C2Y2_list):
    ax.plot(list(range(T)), C2Y2, **p_args, 
            label=r'$\psi$ = {}'.format(psi_list[n]))
    ax.legend()

fig.savefig('habit_persistence_impulse response_CY_alpha=0.1_eta=100.eps', 
            format='eps', 
            dpi=1200)





#==============================================================================
# alpha = .1, eta = 100, psi = .4, .8, 1.6, 2.4 
#==============================================================================
C1Y1_list = []
C2Y2_list = []

psi_list = [.4, .8, 1.6, 2.4 ]


for p in psi_list:
        
    J, A, N1, N2, Ct,_ = solve_habit_persistence(alpha=0.1, eta=100, psi=p)
    
    ## Impulse response for MK   
    C1, Z1 = C_and_Z_path(A, N1, N2, Z0_1, Ct)
    C2, Z2 = C_and_Z_path(A, N1, N2, Z0_2, Ct)

    ## Impulse response for the consumption process C + Y    
    Y1,_ = Y_path(Z1)
    _,Y2 = Y_path(Z2)
    
    C1Y1_list.append(C1 + Y1)
    C2Y2_list.append(C2 + Y2)  


## Impulse response plot for C + Y
fig, axes = create_fig(2,1)

ax = axes[0]
ax.set_ylim(0., 0.5)
for n, C1Y1 in enumerate(C1Y1_list):
    ax.plot(list(range(T)), C1Y1, **p_args, 
            label=r'$\psi$ = {}'.format(psi_list[n]))
    ax.legend()

ax = axes[1]
ax.set_ylim(-0.1, 0.21)
for n, C2Y2 in enumerate(C2Y2_list):
    ax.plot(list(range(T)), C2Y2, **p_args, 
            label=r'$\psi$ = {}'.format(psi_list[n]))
    ax.legend()

fig.savefig('habit_persistence_impulse response_CY_alpha=0.1_eta=100_biger psi.eps', 
            format='eps', 
            dpi=1200)





#%%============================================================================

#==============================================================================
# Functions: Calculate Uncertainty prices
#==============================================================================
"""
Here, we use Su, Sy, Sv, Fy as row vectors for convenience. The counterparts
in the note are (Su)', (Sy)', (Sv)', (Fy)'
"""

# Define Matrix Sy, B, Bx
Sy = np.array([0.704, 0, -0.154])
B = np.hstack([Z0_1, Z0_2])
Bx = B[2:,:]

# Define Fy
Fy = np.array([σ1,σ2])


# Functions
def get_Sv(J, N1, N2, Ut):
    """
    Solve for Su in the note
    
    Output
    =========
    Su: The row vector (Su)' from the note
    
    """
    ##== Calculate Su ==##
    ## Express Ut in terms of Z_{t} and Z_{t+1}
    MKt, MHt, Kt, Ht, X1t, X2t, X2tL1 = symbols('MKt MHt Kt Ht X1t X2t X2tL1')
    MKt1, MHt1, Kt1, Ht1, X1t1, X2t1 = symbols('MKt1 MHt1 Kt1 Ht1 X1t1 X2t1')
    
    T,_ = linear_eq_to_matrix([Ut],
                              MKt, MHt, Kt, Ht, X1t, X2t, X2tL1,
        
                      MKt1, MHt1, Kt1, Ht1, X1t1, X2t1)
    #t1: Ut's coefficient under Z_t
    t1 = T[:7]
    t1 = np.array([t1]).astype(float)
    #t2: Ut's coefficient under Z_{t+1}
    t2 = T[7:]
    t2.append(0)    #X2t is 0 in t2 entry
    t2 = np.array([t2]).astype(float)
    
    ## Get Su
    K = np.vstack([np.hstack([N1,N2]),np.identity(5)])
    JK = np.matmul(J, K)
    
    T1 = np.matmul(t1, K)
    T2 = np.matmul(t2, JK)
    
    Su = T1 + T2
    
    
    ##== Calculate Sv ==##
    """
    We rearranged the equation from the note to get: (Sv)' * A_Sv = b_Sv
    """
    b_Sv = (1 - np.exp(-δ)) * Su + np.exp(-δ) * np.hstack([0,0,Sy])
    A_Sv = (np.identity(5) - np.exp(-δ) * A)
    Sv = np.matmul(b_Sv, np.linalg.inv(A_Sv))
    
    return Sv



def get_SvB(Sv):
    """
    Get the uncertainty price scaled by 1/ξ.
    
    Output
    =========
    SvB: uncertainty price
    
    """
    SvB = np.matmul(Sv, B)
    
    return SvB



def solve_sv(Sv, ξ):
    """
    Solve sv
    
    Output
    ========
    sv: The solution of sv
    
    """
    SvB = get_SvB(Sv)
    sv = ξ/2 * (np.linalg.norm(SvB + np.matmul(Sy, Bx)))**2 / (np.exp(-δ) - 1)
    
    return sv



#==============================================================================
# Uncertainty price for alpha = .1, eta = 60 and psi = .1, .2, .4, .8.  
#==============================================================================
sens_list = []
psi_list = [.1, .2, .4, .8]
for p in psi_list:
    J, A, N1, N2, Ct, Ut = solve_habit_persistence(alpha = 0.1, 
                                                   eta = 60,
                                                   psi = p)
    Sv = get_Sv(J, N1, N2, Ut)
    SvB = get_SvB(Sv)
    SvBFy = SvB + Fy
    SvBFy = [float('%.3g' % x) for x in SvBFy[0]]
    sens_list.append(SvBFy)

for n, p in enumerate(psi_list):  
    print('uncertainty price vector for psi = {0}: {1}'.format(p,sens_list[n]))


#==============================================================================
# Uncertainty price for alpha = .1, eta = 100, psi = 1.6
#==============================================================================
sens_list = []
J, A, N1, N2, Ct, Ut = solve_habit_persistence(alpha = 0.1, 
                                               eta = 100,
                                               psi = 1.6)
Sv = get_Sv(J, N1, N2, Ut)
SvB = get_SvB(Sv)
SvBFy = SvB + Fy
SvBFy = [float('%.3g' % x) for x in SvBFy[0]]


print('The uncertainty price vector for ' +\
      'alpha = 0.1, eta = 100, psi = 1.6: {0}'.format(SvBFy))
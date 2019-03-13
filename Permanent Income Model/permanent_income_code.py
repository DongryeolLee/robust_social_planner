import numpy as np
import numpy.matlib as nm
import numpy.linalg as nl
import matplotlib.pyplot as plt

# Parameters
T = 2000        # Time horizon
S = 2           # Impulse date
ρ = 0.00663     # rate of return on assets
ν = 0.00373     # constant in the log income process
σ1 = 0.108*1.33 # Permanent shock
σ2 = 0.155*1.33 # Transitory shock
Ax = np.array([[0.704,0,0],[0,1,-0.154],[0,1,0]])
DyT = np.array([[0.704,0,0],[0,0,-0.154]])


# =============================================================================
#  2.3: Impulse response -- Y
# =============================================================================
def income_path(T=T, S=S, sigma1=σ1, sigma2=σ2):
    """
    Time path of log income given shock sequence.
    
	Input
	=======
	T: the time horizon, default 2000
	S: Impulse date, default 2
	sigma1: permanent shock, default 0.14364
	sigma2: transitory shock, default 0.20615

    Output
    =======
    Y1: the impulse response path of income regarding the permanent shock
    Y2: the impulse response path of income regarding the transitory shock
    """
    w = nm.zeros((1,T+S))
    X = nm.zeros((3,T+S))
    Y = nm.zeros((2,T+S))
    Bx = np.matrix([[sigma1],[sigma2],[0]])
    FyT = Bx[:2,:]
    w[:,S] = 1
    
    for t in range(1, T+S-1):
        X[:,t+1] = Ax @ X[:,t] + Bx @ w[:,t+1]
        Y[:,t+1] = Y[:,t] + DyT @ X[:,t] + FyT @ w[:,t+1]
    
    Y1 = np.asarray(Y[0,:]).flatten()
    Y2 = np.asarray(Y[1,:]).flatten()
    
    return Y1, Y2



# =============================================================================
# 2.3： Impulse response -- C
# =============================================================================
def consumption_income_ratio_path(k_bar=0, T=T, S=S, rho=ρ, nu=ν, sigma1=σ1, sigma2=σ2):
    """
    Time path of log consumption-income ratio given shock sequence
     
	Input
	=======
    k_bar: steady state capital, default 0
	T: the time horizon, default 2000
	S: Impulse date, default 2
	rho: the asset return, default 0.00663
	nu: the constant in the logarithm of income process, default 0.00373
	sigma1: permanent shock, default 0.14364
	sigma2: transitory shock, default 0.20615
   
    Output
    =======
    C1: the impulse response path of log consumption-income ratio regarding 
        the permanent shock
    C2: the impulse response path of log consumption-income ratio regarding 
        the transitory shock
    """
    w = nm.zeros((1,T+S))
    X = nm.zeros((3,T+S))
    C = nm.zeros((2,T+S))   
    Bx = np.matrix([[sigma1],[sigma2],[0]])
    FyT = Bx[:2,:]
    w[:,S] = 1
    
    lam = np.exp(nu - rho)
    c_bar = np.log((np.exp(rho) - np.exp(nu)) * k_bar + 1)
    G = np.exp(rho - c_bar) - np.exp(nu - c_bar)
    I = nm.eye(3)
    M = lam * DyT @ nl.inv(I - lam * Ax)
    M_star = M * (1 + k_bar * G)
    
    for t in range(1, T+S-1):
        X[:,t+1] = Ax @ X[:,t] + Bx @ w[:,t+1]
        dY =  DyT @ X[:,t] + FyT @ w[:,t+1]
        C[:,t+1] = C[:,t] - k_bar * G * dY - (1/lam) * M_star @ X[:,t] + M_star @ X[:,t+1]
        
    C1 = np.asarray(C[0,:]).flatten()
    C2 = np.asarray(C[1,:]).flatten()
    
    return C1, C2



# =============================================================================
# 2.3： Impulse response -- C + Y
# =============================================================================
def consumption_path(k_bar=0, T=T, S=S, rho=ρ, nu=ν, sigma1=σ1, sigma2=σ2):
    """
    Time path of log consumption-income ratio given shock sequence
     
	Input
	=======
    k_bar: steday state capital, default 0
	rho: the asset return, default 0.00663
	nu: the constant in the logarithm of income process, default 0.00373
	T: the time horizon, default 2000
	S: Impulse date, default 2
	sigma1: permanent shock, default 0.14364
	sigma2: transitory shock, default 0.20615
   
    Output
    =======
    C1Y1: the impulse response path of log consumption regarding the permanent shock
    C2Y2: the impulse response path of log consumption regarding the transitory shock
    """
    C1, C2 = consumption_income_ratio_path(k_bar=k_bar, T=T, S=S, rho=rho, nu=nu, 
                                           sigma1=sigma1, sigma2=sigma2)
    Y1, Y2 = income_path(T=T, S=S, sigma1=sigma1, sigma2=sigma2)
    
    C1Y1 = C1 + Y1
    C2Y2 = C2 + Y2
    
    return C1Y1, C2Y2


# =============================================================================
# Plot impulse responses (non-interactive)
# =============================================================================
def plot_responses(R1, R2, T=T, S=S):
    """
    Time path of log consumption-income ratio given shock sequence
     
	Input
	=======
    R1: permanent shock responses
	R2: transitory shock responses
	T: the time horizon, default 2000
	S: Impulse date, default 2
   
    Output
    =======
    The figure of responses
    """
    # Plot income responses
    fig, axes = plt.subplots(2, 1,figsize=(8,8))
    plt.subplots_adjust(hspace=0.5)
    p_args = {'lw': 2, 'alpha': 0.7}

    for ax in axes:
        ax.grid(alpha=0.5)
        ax.set_xlim(0,40)
        ax.set_xlabel(r'Quarters')
        ax.set_ylim(0, 0.6)

    ax = axes[0]
    ax.plot(list(range(T)), R1[S:], 'g-', **p_args) 

    ax = axes[1]
    ax.plot(list(range(T)), R2[S:], 'b-', **p_args)

    return
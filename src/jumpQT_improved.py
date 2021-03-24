# Improved version of jumpQT (several init values can be tested), Stochastic Schroedinger Equation With Jumps

import qutip as qt
import numpy as np
import matplotlib.pyplot as plt
import random as rnd

# some key variables, N: timesteps, T: final time, dt: timestep
# -------------------------
N = 100
T = 10
dt = T/N
theta = np.sqrt(dt)     # interaction strength parameter, relating interaction time and lindblad rates

times = np.linspace(0,T,N) # array of times used when plotting results


# randomization properties
# -------------------------
seed    = 1300
rng     = rnd.seed(seed)


# initial
bas0 = qt.basis(2,0) # zero ket
bas1 = qt.basis(2,1) # one ket

bas1_np = np.array([[0],[1]])
print(bas1_np)
test = (bas1*bas1.dag() - 2.) * bas1_np
print(test)

#beta0 = 1/np.sqrt(2)                           #only relevant for single case
#alpha0 = np.sqrt(1 - np.abs(beta0*beta0))


# setting up initial states
# -------------------------
N_states        = 9     # number of states to simulate

beta_max        = 0.9   # absolute squared value of beta
beta_min        = 0.1
beta_values     = np.sqrt( np.linspace(beta_min, beta_max, N_states) )

alpha_values    = np.sqrt(1 - np.abs(beta_values*beta_values))

psi_init        = np.zeros((N_states,2))
psi_init[:,0]   = alpha_values
psi_init[:,1]   = beta_values

print("initial states:\n", psi_init)



# evolution array storing SSE evolved states
# -------------------------
psi_evol    = np.zeros((N,N_states,2)) # array to store evolution of state
psi_evol[0] = psi_init

def SSE_CNOT(psi, beta0):
    
    p1      = (theta*theta) * np.abs(psi[1]*psi[1])
    #p1      = (theta*theta) * beta0*beta0
    p       = rnd.random()

    if p > p1:
        psi_ = psi - (theta*theta/2) * ( bas1*bas1.dag() - (  np.abs(psi[1])**2 )) * psi
    
    else:
        psi_ = psi  + ( (bas1*bas1.dag())/(np.abs(psi[1])) - qt.qeye(2) ) * psi
    

    return psi_


# Simulating trajectories for the N_states
# -------------------------
for i in range(N-1):
    for j in range(N_states):
        psi_evol[i+1,j] = SSE_CNOT(psi_evol[i,j], psi_evol[0,j,1])



# Plotting |beta|**2 for the N_states
# -------------------------
for beta in range(N_states):
    abs_beta_squared = np.abs(psi_evol[:,beta,1]*psi_evol[:,beta,1])
    plt.plot(times, abs_beta_squared, label=str(psi_evol[0,beta,1]**2))

#plt.legend()
plt.ylabel("|beta|^2")
plt.xlabel("time")
plt.show()
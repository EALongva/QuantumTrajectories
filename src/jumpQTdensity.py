# jumpQTdensity, Stochastic Schroedinger Equation With Jumps used to approximate density matrix of system

import qutip as qt
import numpy as np
import matplotlib.pyplot as plt
import random as rnd

# Functions
# -------------------------

def SSE_CNOT_NSTATES(N, T, N_states, beta_max, beta_min, seed):
    # Function derived from the file jumpQT_improved

    rng                 = rnd.seed(seed)

    bas0                = qt.basis(2,0) # zero ket
    bas1                = qt.basis(2,1) # one ket

    dt                  = T/N
    theta               = np.sqrt(dt)     # interaction strength parameter, relating interaction time and lindblad rates
    times               = np.linspace(0,T,N) # array of times used when plotting results

    beta_values         = np.sqrt( np.linspace(beta_min, beta_max, N_states) )
    alpha_values        = np.sqrt(1 - np.abs(beta_values*beta_values))

    psi_init            = np.zeros((N_states,2))
    psi_init[:,0]       = alpha_values
    psi_init[:,1]       = beta_values

    psi_evol            = np.zeros((N,N_states,2)) # array to store evolution of state
    psi_evol[0]         = psi_init

    for i in range(N-1):

        for j in range(N_states):
            
            p1      = (theta*theta) * np.abs(psi_evol[i,j,1]*psi_evol[i,j,1])
            p       = rnd.random()

            if p > p1:
                psi_evol[i+1,j] = psi_evol[i,j] - (theta*theta/2) * ( bas1*bas1.dag() - (  np.abs(psi_evol[i,j,1])**2 )) * psi_evol[i,j]
    
            else:
                psi_evol[i+1,j] = psi_evol[i,j]  + ( (bas1*bas1.dag())/(np.abs(psi_evol[i,j,1])) - qt.qeye(2) ) * psi_evol[i,j]

    return psi_evol

# some key variables, N: timesteps, T: final time, dt: timestep
# -------------------------
N               = 100
T               = 10
times           = np.linspace(0,T,N) # array of times used when plotting results
N_states        = 3
beta_max        = 0.6
beta_min        = 0.4


# Master array to hold a number S of simulations of Stochastic SE with jumps
# -------------------------
S               = 100
MasterArray     = np.zeros((S,N,N_states,2))


# Computing trajectories with varying seed
# -------------------------
base_seed = 1337

for k in range(S):
    MasterArray[k] = SSE_CNOT_NSTATES(N, T, N_states, beta_max, beta_min, base_seed+k)

meanEvolution = np.mean(MasterArray, axis=0)
#print(meanEvolution)



# Plotting |beta|**2 for the N_states (sample 1)
# -------------------------
psi_evol = meanEvolution
for beta in range(N_states):
    abs_beta_squared = np.abs(psi_evol[:,beta,1]*psi_evol[:,beta,1])
    plt.plot(times, abs_beta_squared, label=str(psi_evol[0,beta,1]**2))

#plt.legend()
plt.ylabel("|beta|^2")
plt.xlabel("time")
plt.show()


# note: could speed up algorithm by breaking trajectories that have jumped (right now these are recalculated) 
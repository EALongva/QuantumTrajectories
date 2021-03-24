# jumpQTdensity, Stochastic Schroedinger Equation With Jumps used to approximate density matrix of system

import qutip as qt
import numpy as np
import matplotlib.pyplot as plt
import random as rnd

# some key variables
N = 100
T = 10
dt = T/N
theta = np.sqrt(dt)     # still not entirely convinced this is correct

times = np.linspace(0,T,N)

# randomization properties
seed    = 1329
rng     = rnd.seed(seed)

# initial
bas0 = qt.basis(2,0) # zero ket
bas1 = qt.basis(2,1) # one ket
beta0 = 1/np.sqrt(2)
alpha0 = np.sqrt(1 - np.abs(beta0*beta0))
print(alpha0)

# setting up initial states
# -------------------------
N_states    = 10    # number of states to simulate

beta_max    = 0.9   # absolute squared value of beta
beta_min    = 0.1
beta_values = np.sqrt( np.linspace(beta_min, beta_max, N_states) )

print(beta_values*beta_values)



"""
psi0 = alpha0*bas0 + beta0*bas1

psin    = np.zeros((N,2,1)) # array to store evolution of state
psin[0] = psi0

def SSE_CNOT(psi):
    
    #p1      = (theta*theta) * np.abs(psi[1][0]*psi[1][0])
    p1      = (theta*theta) * beta0*beta0
    p       = rnd.random()

    #print(psi[1][0])

    #print(p1)

    if p > p1:
        psi_ = psi - (theta*theta/2) * ( bas1*bas1.dag() - (  np.abs(psi[1][0])**2 )) * psi
    
    else:
        psi_ = psi  + ( (bas1*bas1.dag())/(np.abs(psi[1][0])) - qt.qeye(2) ) * psi
    

    return psi_



for i in range(N-1):
    psin[i+1] = SSE_CNOT(psin[i])

alpha2  = np.abs(psin[::,0])**2
beta2   = np.abs(psin[::,1])**2


plt.plot(times, beta2, label="|beta|^2", c="red")
#plt.plot(times, alpha2, label="|alpha|^2", c="green")
plt.legend()
plt.show()

"""
# Stochastic Schroedinger Equation With Jumps

import qutip as qt
import numpy as np
import matplotlib.pyplot as plt
import random as rnd

# some key variables
N = 100
T = 10
dt = T/N
theta = np.sqrt(dt)     # still not entirely convinced this is correct

# randomization properties
seed    = 1337
rng     = rnd.seed(seed)

# initial
bas0 = qt.basis(2,0) # zero ket
bas1 = qt.basis(2,1) # one ket
psi0 = ( bas0 + bas1 )/np.sqrt(2)

expect = bas1.dag()*psi0
#print((bas1.dag()*psi0).norm()**2)

#print(psi0)

psin    = np.zeros((N,2,1)) # array to store evolution of state
psin[0] = psi0

""" not sure about this part """

n = np.linspace(0,T,N)

alpha = 1/np.sqrt(2)
beta = 1/np.sqrt(2)

denom = np.abs(alpha*alpha) + np.abs(beta*beta)*np.exp(-n*theta*theta)

alphasqr_n = np.abs(alpha*alpha) / denom
betasqr_n = np.abs(beta*beta)*np.exp(-n*theta*theta) / denom

""" """


def SSE_CNOT(psi):
    
    p1      = (theta*theta) * np.abs(beta*beta)
    p       = rnd.random()

    if p > p1:
        psi_ = psi - (theta*theta/2) * ( bas1*bas1.dag() - ((bas1.dag()*psi0)).norm()**2 ) * psi
    else:
        psi_ = psi + ( (bas1*bas1.dag())/((bas1.dag()*psi0)).norm() - qt.qeye(2) ) * psi

    return psi_


"""
plt.plot(n, alphasqr_n, n, betasqr_n)
plt.show()
"""

for i in range(N-1):
    psin[i+1] = SSE_CNOT(psin[i])

alpha2  = np.abs(psin[::,0])**2
beta2   = np.abs(psin[::,1])**2

plt.plot(n, beta2, label="|beta|^2", c="red")
plt.plot(n, alpha2, label="|alpha|^2", c="green")
plt.legend()
plt.show()
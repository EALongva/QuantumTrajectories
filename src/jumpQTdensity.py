# jumpQTdensity, Stochastic Schroedinger Equation With Jumps used to approximate density matrix of system

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
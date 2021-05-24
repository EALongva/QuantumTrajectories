# testing to find validity of results from jumpQTdensity.py

import qutip as qp
import numpy as np
import matplotlib.pyplot as plt

times     = np.load("../dat/SSEjump/QTextdrive_N_1500ntraj_10000_swap_times.npy")
dt        = times[-1]/times.size

theta = np.sqrt(dt)

bas0 = np.matrix([[1 + 0j], [0]])
bas1 = np.matrix([[0],[1 + 0j]])

# Hamiltonian external driving

omega0  = 1.0
omega1  = 1.0
omega   = 0.1

sigx = np.matrix([ [0,1], [1,0] ])
sigy = np.matrix([ [0,-1j], [1j,0] ])
sigz = np.matrix([ [1,0], [0,-1] ])

H = 0.5*(omega0 - omega)*qp.sigmaz() + 0.5*omega1*qp.sigmax()


A_cnot = qp.basis(2,1)*qp.basis(2,1).dag()
A_swap = qp.basis(2,0)*qp.basis(2,1).dag()

a = 1.0
A = A_swap.copy()

c_ops = [a*A]

psi0  = (qp.basis(2,0) + qp.basis(2,1))/np.sqrt(2) # init state


result = qp.mesolve(H, psi0, times, c_ops, [])

master = np.zeros((times.size,2,1))
master[:,0,0] = np.array(result.states)[:,0,0]
master[:,1,0] = np.array(result.states)[:,1,1]

filename_master = "../dat/SSEjump/" + "LBextdrive_" + "N_" + str(times.size-1) + "_swap"
np.save(filename_master, master)
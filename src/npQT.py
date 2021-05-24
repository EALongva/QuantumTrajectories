# testing a numpy only program for solving the stochastic schroedinger equation (63) in Bruns article

import matplotlib.pyplot as plt
import numpy as np
import qutip as qp
import random as rnd
import time as time
from scipy.linalg import expm


def SSE_solver(H, psi0, A, a, N, T, theta=1, ntraj=1, baseseed=1337, debug=False):

    # H: Hamiltonian (TLS -> 2 by 2 matrix), psi0: initial state (2, column array),
    # A: "base" LB operator (2 by 2 matrix), a: corresponds to gamma factors (float)
    # N: number of timesteps per trajectory simulation, dt: timestep
    # theta: strength of interaction (float), ntraj: number of simulated trajectories to average over

    dt = T/N

    if theta == 1:
        theta = np.sqrt(dt)     # default theta value

    
    I = np.matrix( [[1,0], [0,1] ])
    L = a*A # LB operator
    LL = L.H.dot(L)

    print("Initial state: ", psi0)
    print("theta value: ", theta)
    print("Lindblad operator: \n", L)
    print("Hamiltonian: \n", H)

    times = np.linspace(0,T,N+1)

    masterArray = np.zeros((ntraj, N+1, 2, 1), dtype=psi0.dtype) # must have shape (2,1) as base for each state ket (else lose distinction between column and row vectors)
    
    for n in range(ntraj):

        if debug and n<1:
            print('debug on, timing first trajectory simulation ... ')
            ping0 = time.perf_counter()

        if debug and n>=1:
            ping0 = time.perf_counter()

        rnd.seed(baseseed+n)

        states = []
        states.append(psi0)

        for i in range(N):

            expval = (states[i].H @ LL @ states[i])[0,0].real # [0,0].real is not necessary but makes things cleaner

            dN = 0
            p1 = expval*dt

            if rnd.random() <= p1:
                dN = 1

            newstate = states[i] - (dt/2) * ( LL @ states[i] - expval * states[i] ) + ( L/np.sqrt(expval) - I) @ states[i] * dN

            states.append(newstate)
                    
            states[i+1] = expm(-1j*H*dt) @ states[i+1] # standard Hamiltonian time evolution

        masterArray[n,:,:,:] = states

        if debug and n<1:
            ping1 = time.perf_counter()
            simtime = ping1 - ping0
            print('time of first trajectory sim: ', simtime )
            print('projected total sim time: ', simtime*ntraj)
            print('estimated finish time: ', time.ctime(time.time()+1.2*simtime*(ntraj-1)))

        if debug and n>1:
            ping1 = time.perf_counter()
            simtime = ping1 - ping0
            print('time of ', n, ' trajectory sim: ', simtime )
            print('estimated finish time: ', time.ctime(time.time()+simtime*(ntraj-n)))

    
    return times, masterArray


# general parameters

# testing bra and ket notation with numpy

bas0 = np.matrix([[1 + 0j], [0]])
bas1 = np.matrix([[0],[1 + 0j]])

# Lindblad operators

A_cnot = bas1.dot(bas1.H)
A_swap = bas0.dot(bas1.H)

# Pauli matrices

sigx = np.matrix([ [0,1], [1,0] ])
sigy = np.matrix([ [0,-1j], [1j,0] ])
sigz = np.matrix([ [1,0], [0,-1] ])
iden = np.eye(2)

# Hamiltonian external driving
omega0  = 1.0
omega1  = 1.0
omega   = 0.1

def H_extDrive(t):
    return 0.5*omega0*sigz + 0.5*omega1*(np.cos(omega*t)*sigx + np.sin(omega*t)*sigy)

H_extDriveT = 0.5*(omega0 - omega)*sigz + 0.5*omega1*sigx


# staging

a = 1.0
A = np.matrix.copy(A_cnot)
N = 1500
T = 15
theta = np.sqrt(T/N)
ntraj = 10000
seed = 301852

psi0    = (bas0+bas1)/np.sqrt(2)

# H, psi0, A, a, N, dt, theta=1, ntraj=1, baseseed=1337
times, master = SSE_solver(H_extDriveT, psi0, A, a, N, T, theta=theta, ntraj=ntraj, baseseed=seed, debug=True)


filename_master = "../dat/SSEjump/" + "QTextdrive_" + "N_" + str(N) + "ntraj_" + str(ntraj) + "_cnot"
np.save(filename_master, master)

filename_times = "../dat/SSEjump/" + "QTextdrive_" + "N_" + str(N) + "ntraj_" + str(ntraj) + "_cnot" + "_times"
np.save(filename_times, times)


"""
plt.plot(times, np.conj(master[0,:,0,0])*master[0,:,0,0], label='alpha')
plt.plot(times, np.conj(master[0,:,1,0])*master[0,:,1,0], label='beta')
plt.legend()
plt.show()
"""

# testing matrix multiplication with numpy

"""
print("init state and hconjugate")
print(psi0)
print(psi0.H)
print("LB operator and hconjugate")
print(A)
print(A.H)

AA = A.H @ A

expval = (psi0.H @ A.H @ A @ psi0).real

expval1 = (psi0.H @ AA @ psi0).real

print("comptued expectation value")
print(expval)
print(expval1)
#print(expval.shape)

if expval > 0.1:
    print("yo")
"""

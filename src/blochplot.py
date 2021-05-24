# testing to find validity of results from jumpQTdensity.py

import numpy as np
import qutip as qt
import matplotlib.pyplot as plt

QTmaster    = np.load("../dat/SSEjump/QTextdrive_N_1500ntraj_10000_cnot.npy")
times       = np.load("../dat/SSEjump/QTextdrive_N_1500ntraj_10000_cnot_times.npy")

def dag(x):
    # hermitian conjugate of 3 dim array with (2,1) matrixes along first axis
    return np.conj(np.transpose(x, (0,2,1)))

# computing the QT solutions bloch representation (stereographic projection)
"""
qtMean = np.mean(QTmaster, axis=0)

ux = (qtMean[:,1,0] / qtMean[:,0,0]).real
uy = (qtMean[:,1,0] / qtMean[:,0,0]).imag

Px = 2*ux / (1 + ux**2 + uy**2)
Py = 2*uy / (1 + ux**2 + uy**2)
Pz = (1 - ux**2 - uy**2) / (1 + ux**2 + uy**2)

P = [Px, Py, Pz]
"""
# making another attempt

qtMean = np.mean(QTmaster, axis=0)

qtRho = qtMean @ dag(qtMean)

px = qtRho[:,1,0] + qtRho[:,0,1]
py = 1j*(qtRho[:,1,0] - qtRho[:,0,1])
pz = qtRho[:,0,0] - qtRho[:,1,1]

P = [px, py, pz]

# computing LB solution using qutip

omega0  = 1.0
omega1  = 1.0
omega   = 0.1

sigx = np.matrix([ [0,1], [1,0] ])
sigy = np.matrix([ [0,-1j], [1j,0] ])
sigz = np.matrix([ [1,0], [0,-1] ])

H = 0.5*(omega0 - omega)*qt.sigmaz() + 0.5*omega1*qt.sigmax()

A_cnot = qt.basis(2,1)*qt.basis(2,1).dag()
A_swap = qt.basis(2,0)*qt.basis(2,1).dag()

a = 1.0
A = A_cnot.copy()

c_ops = [a*A]

psi0  = (qt.basis(2,0) + qt.basis(2,1))/np.sqrt(2) # init state


result = qt.mesolve(H, psi0, times, c_ops, [])

rho = np.array(result.states)

rx = rho[:,1,0] + rho[:,0,1]
ry = 1j*(rho[:,1,0] - rho[:,0,1])
rz = rho[:,0,0] - rho[:,1,1]

R = [rx, ry, rz]


# plotting using bloch sphere
"""
b = qt.Bloch()
b.font_size = 8
b.point_size = [1,1,1,1]
b.view = [-60,30]

b.make_sphere()
b.add_points(P)
b.add_points(R)
b.fig.legend(['QT', 'LB'])
b.render()

filename = "../fig/SSEjump/QTextdrive_bloch_cnot.png"
b.fig.savefig(filename, dpi=400)
"""

# plotting in standard matplotlib

xyz = ['x', 'y', 'z']
alpha = [0.9, 0.7, 0.5]
lines = ['-', '--', '-.']

for p, u, a, l in zip(P, xyz, alpha, lines):
    plt.plot(times, p, 'r' + l, label='QT' + u, alpha=a)

for r, u, a, l in zip(R, xyz, alpha, lines):
    plt.plot(times, r, 'g' + l, label='LB' + u, alpha=a)

plt.legend()
plt.grid()
plt.xlabel('time')
plt.ylabel('bloch vector components')
plt.title('Bloch vector comparison\nQuantum Trajectories vs Lindblad solution')
plt.show()
# testing to find validity of results from jumpQTdensity.py

import numpy as np
import matplotlib.pyplot as plt

QTmaster    = np.load("../dat/SSEjump/QTextdrive_N_1500ntraj_10000_swap.npy")
times       = np.load("../dat/SSEjump/QTextdrive_N_1500ntraj_10000_swap_times.npy")

LBmaster    = np.load("../dat/SSEjump/LBextdrive_N_1500_swap.npy")


QTmaster_SQRD_MEAN = np.mean(np.conj(QTmaster) * QTmaster, axis=0)

#print(QTmaster_SQRD_MEAN.real)

plt.title('Mean of Quantum Trajectories solution vs Lindblad solution \n for TLS with external drive Hamiltonian ' + r'$\omega_0 = \omega_1 = 1$' + '\nand ' + r'$\omega = 0.1$' + ' SSE with jumps (swap)')
plt.plot(times, QTmaster_SQRD_MEAN[:,1], label='QTrajectories')
plt.plot(times, LBmaster[:,1], label='Lindblad')
plt.xlabel(r'$t$' + ' [unitless time]')
plt.ylabel(r'$|\beta|^2$' + ' [probability amplitude]')
plt.legend()
plt.tight_layout()

figname = '../fig/SSEjump/QTvsLBplot' + '_N_' + str(times.size-1) + '_ntraj_' + str(QTmaster[:,0,0,0].size) + '_swap.png'
plt.savefig(figname, dpi=400)

plt.show()
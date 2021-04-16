# testing to find validity of results from jumpQTdensity.py

import qutip as qp
import numpy as np
import matplotlib.pyplot as plt

N               = 100
T               = 10
times           = np.linspace(0,T,N) # array of times used when plotting results
dt              = T/N

theta = np.sqrt(dt)

delta           = 0.0
episilon        = 0.0 # only interested in simplest Hamiltonian for the TLS now

H               = delta*qp.sigmaz() + episilon*qp.sigmax()

times           = np.linspace(0,T,N) # array of times used when plotting results

psi0            = (qp.basis(2,0) + qp.basis(2,1))/np.sqrt(2)
#psi0 = qp.basis(2,0)
print(psi0)

sm = qp.basis(2,1)*qp.basis(2,1).dag()
print(sm)
sp = sm.dag()
gm = np.sqrt(theta*theta/dt)
gp = np.sqrt(dt)

result = qp.mesolve(H, psi0, times, [gm*sm, gp*sp], [])

print(result.states)
#print(result)

#expect = np.asarray(result.states)[:,0,1]*np.conj(np.asarray(result.states)[:,0,1])

#print(expect)

#plt.plot(result.times, np.asarray(result.states)[:,0,1])
#plt.plot(result.times, result.expect[0])
#plt.plot(result.times, result.expect[1])
#plt.plot(result.times, result.expect[2])
plt.show()
#plt.close
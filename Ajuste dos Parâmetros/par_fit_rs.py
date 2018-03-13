'''
	Fits mu and R parameters for GL neuron
'''

import numpy as np
import matplotlib.pyplot as plt
from random import random as rnd
from joblib import Parallel, delayed
import multiprocessing
import glob

I = np.arange(0, 5, 0.1)   # Current in nA

'''
	F-I curves for the HH neurons.
'''
Frs = np.genfromtxt('../FI modelos HH/fi_modelohh.dat', delimiter  = ',', usecols=(0) )

'''
	A, B, C : Phi(V) parameters
	V : Membrane Potential
'''
def phi_v(V0, k, C, V):
	return 1.0 / ( 1 + np.exp(-k*(V-V0)) )


def simulation(mu, I, R, tau_ref, Vreset):

	# Simulation Parameters
	dt   = 0.1
	tsim = 1000.0
	time = np.arange(0, tsim, dt)
	trefact = 0

	# Phi parameters
	V0 = -35.50
	k = 1.79
	C = 1.0

	# Neuron Parameters
	Vr = -65.0
	V  = np.ones(int(tsim/dt))*Vr   # Initial membrane potential

	frequency = 0
	X = 0

	for i in range(0, len(V)-1):
		if X == 1:
			V[i+1] = Vreset
			if time[i] - trefact >= tau_ref:
				X = 0
		else:
			V[i+1] = mu*V[i] + (1-mu)*Vr + (1-mu)*R*I
			if rnd() < phi_v(V0,k, C, V[i+1]):
				trefact = time[i]
				X = 1
				frequency += 1

	return frequency

def Sweep(mu, R, tau_ref, Vreset):

	Frequency  = []

	for corrente in I:
		Frequency.append( simulation(mu, corrente, R, tau_ref, Vreset) )
	Q = np.mean((Frs - np.array(Frequency))**2)
	return np.array([Q, mu, R, tau_ref, Vreset]).T

###################################################################################################################################

Mu         = [0.99]#np.arange(0.90, 1.01, 0.01)
Resistence = [43.]#np.arange(40, 50.0, 1)
tau        = [1.80]#np.arange(1.5, 2.1, 0.1)
Vreset     = [-74.30]#np.arange(-80.0, -55.0, 5)

num_cores = multiprocessing.cpu_count()
Par  = np.squeeze( Parallel(n_jobs=num_cores)(delayed(Sweep)(mu, R, tau_ref, vres) for mu in Mu for R in Resistence for tau_ref in tau for vres in Vreset) )

idx = np.argmin(Par[:,0])

bestMu = Par[idx, 1]
bestR  = Par[idx, 2]
bestTau  = Par[idx, 3]
bestVreset = Par[idx, 4]

f = []
for corrente in I:
	f.append(simulation(bestMu, corrente, bestR, bestTau, bestVreset))

plt.plot(I, Frs)
plt.plot(I, f)
plt.legend(['modelo HH RS', 'modelo GL FS'])
plt.savefig('rs_hh_fi.pdf', dpi = 600)

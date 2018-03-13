	#GL MODEL

from brian2 import *
import numpy as np
import multiprocessing
from joblib import Parallel, delayed
import pandas as pd

def phiv(V, gamma=0.037, vt=-64.5, vs=-37.5, r=3.0):
	r'''
		Evaluate the firing probability for a given membrane potential value V
	'''
	if V < vt:
		return 0
	elif V>=vt and V<vs:
		return (gamma*(V-vt))**r
	elif V>= vs:
		return 1

def SimulateGL(Tsim = 200, dt = 1.0):
	r'''
		dt: Numerical integration time step [ms]
		TSim: Simulation time [ms]
	'''
	Npoints = int(Tsim / dt) # Number of steps

	R    = 100.0        # Resistence [Gohm]
	taum = 20.0         # Membrane time constant [ms]
	vr   = -65.0        # Reset potential [mV]
	vt   = -64.5        # Threshold potential [mV]
	vs   = -37.5        # Saturation potential [mV]
	I    = 1.0          # Input current [pA]
	# Firing probability parameters
	gamma = 0.037
	r = 3.0

	# Time vector
	Time = np.linspace(0, Tsim, Npoints)
	# Membrane otential vector
	v    = np.ones(Npoints)*vr
	# Record membrne potential at spike time
	vspike = []
	tspike = []
	X = 0
	# Integration using Euler's method
	for i in range(Npoints-1):
		if X == 1:
			v[i+1] = vr
			X = 0
		elif X == 0:
			v[i+1] = v[i] + (dt/taum) * ( -v[i] + vr + R*I )
			# Check for spikes
			if np.random.rand() < phiv(v[i+1]):
				vspike.append(v[i+1])
				tspike.append(Time[i+1])
				X = 1
	return Time, v, tspike, vspike

def returnPhi(vm, Tsim = 1e6, delta_t = 0.1):

	t, v, ts, vth = SimulateGL(Tsim = Tsim, dt = delta_t)

	r'''
		\Phi(V) reconstruction
	'''
	# Membrane potential reconstrution
	n1, x1 = np.histogram(v, vm)
	# Thresholds Histogram
	n2, x2 = np.histogram(vth, vm)

	# Recovered \Phi(V)
	phi_exp = n2.astype(np.double) / n1.astype(np.double)

	#a1 = vm[~np.isnan(phi_exp)]
	#a2 = phi_exp[~np.isnan(phi_exp)]

	#np.savetxt(filename, np.array([a1, a2]).T, delimiter = ',')
	return phi_exp

def retunrMeanPhi(vm, k = 10, Tsim = 1e6, delta_t = 0.1):
	filename = 'data/Tsim_' + str(int(Tsim)) + '_dt_' + str(int(10*delta_t)) + '.csv'
	pe = np.zeros(len(vm)-1)
	data = pd.DataFrame()
	for i in range(k):
		phi_exp = returnPhi(vm, Tsim = Tsim, delta_t = delta_t)
		data[str(i)] = phi_exp

	pe_mean = data.mean(axis = 1, skipna = True).values
	pe_std  = data.std(axis = 1, skipna = True).values
	
	a1 = vm[~np.isnan(pe_mean)]
	a2 = pe_mean[~np.isnan(pe_mean)]
	a3 = pe_std[~np.isnan(pe_mean)]
	a4 = [phiv(x) for x in a1]
	np.savetxt(filename, np.array([a1, a2, a3, a4]).T, delimiter = ',')


vm = np.arange(-65.0, -35, 0.1)

delta_t = np.linspace(0.1, 1, 4)
tsim   = np.array([1e6])

Parallel(n_jobs=40)(delayed(retunrMeanPhi)(vm, k = 10, Tsim = t, delta_t = dt) for t in tsim for dt in delta_t)


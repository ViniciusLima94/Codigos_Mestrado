# Routine to determine the probability of emiting a spike, from neuronal data (Membrane potential curve)
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from operator import truediv
import matplotlib.pyplot as plt
from detect_peaks import detect_peaks

neuron_type = 'FS'

# Reading data
data_matrix = pd.read_csv('data'+neuron_type+'/'+neuron_type+'trace.dat', delimiter=",", header=None).as_matrix()
data_matrix = np.squeeze(data_matrix) # Matrix to numpy array

# Index of the peaks
index = detect_peaks(data_matrix, mph = -10)  # mph = -10, peaks with height greater than -10mV
# Store number of peaks in the signal
Npeaks = len(index)
# Store threshold values
thr_values = []
	
# Store all potential values except the ones above the threshold value for the
# respective peak
data_matrix2 = []
t = np.arange(0,5.05,0.05)

# Finding thresholds
plt.figure()
for i in range(0,Npeaks-1):
	# Separate each peak, to find the threshold
        # P is one peak of a given experimente
        # P is choosed from the peak to 1.5ms ago
	P = data_matrix[index[i]-60:index[i]].copy()
	# dP/dt
	P1 = np.diff(P,1)
	#(d²P/dt²)
	P2 = np.diff(P,2)
	# Method VII
	Kp = P2 * (1 + P1[:-1]**2)**-1.5
	# Find the max Kp index
	kp_max_idx = np.argmax(Kp)+1
	# Append threshold found in thr_values
	thr_values.append(P[kp_max_idx].copy())
	#if i < 10:
	#	plt.plot(P, 'k')
	#	plt.plot(kp_max_idx, thr_values[i], 'ro')
	# Overwriting P
	aux = data_matrix[index[i]-60:index[i+1]-59].copy()
	aux[aux > thr_values[i]] = np.nan
	# Append peak in data_matrix2, excluding all values
	# greater than the found threshold
	data_matrix2 = np.concatenate((data_matrix2, aux.copy()), axis = 0)
#plt.savefig(neuron_type+'_trace.pdf', dpi = 600)
bl = 0.2 # Length of bins
v_m = np.arange(-80,-30,bl)
data_matrix2 = data_matrix2[~np.isnan(data_matrix2)] # Removing NaN

n1, x1 = np.histogram(data_matrix2, v_m)
n2, x2 = np.histogram(thr_values, v_m)
# Finding non-zero elements to avoid division by zero
#idx = np.nonzero(n2)

# Probability
p = n2.astype(np.double) / n1.astype(np.double)

v = v_m[~np.isnan(p)]
phi = p[~np.isnan(p)]

# Save Histograms
plt.figure()
plt.hist(data_matrix2, bins = v_m, alpha = 0.8, log=True)
plt.hist(thr_values, bins = v_m, alpha = 0.8, log=True)
plt.savefig(neuron_type+'_hist.pdf', dpi = 600)
'''
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

h1 = ax1.bar(x1[:-1], n1, color='blue')
h2 = ax2.bar(x2[:-1],n2, color='black')
ax1.legend((h1[0],h2[0]) , ('Vm', 'Vth'), loc='upper left')
ax1.set_xlabel('Membrane Potential [mV]')
ax1.set_ylabel('Count')
ax2.set_ylabel('Count')
plt.tight_layout()
plt.savefig('RShist.pdf', dpi = 600)
'''

'''
plt.figure()
plt.plot(v_m[:-1], p, 'o')
plt.ylabel(r'$\Phi$')
plt.xlabel('Membrane Potential [mV]')
plt.xlim((-40,-33))
plt.savefig('RSphi.pdf', dpi = 600)
'''

np.savetxt(neuron_type+'phi.dat', np.array([v, phi]).T, delimiter = ',')

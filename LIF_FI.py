# -*- coding: utf-8 -*-
from brian2 import *
import pandas as pd
import sys  
reload(sys)  
sys.setdefaultencoding('utf8')

defaultclock.dt = 0.1 * ms


def LIF(taum = 20, tau_ref = 2.0, Vrest =  0, Vreset = 10, Vth = 20, R = 1):

	r'''
	  Neuron Parameters
	  tau_m : Membrane time constant
	  tau_ref : Refractory period
	  Vrest   : Resting potential
	  Vreset  : Reset potential
	  Vth     : Threshold
	  eqs     : Model equation
	  v_ext   : External input frequency
	'''
	tau_m    =  taum * ms
	tau_ref  =  tau_ref * ms
	Vrest    =  Vrest * mV
	Vreset   =  Vreset * mV
	Vth      =  Vth * mV

	C, k, V0 = 1.0, 1.79 * 1/mV, -35.50 * mV

	eqs = '''
		dV/dt = -(V - Vrest - R*I) / tau_m : volt (unless refractory)
		I : volt
		'''

	Neurons = NeuronGroup(1000, model = eqs, threshold = 'V>Vth', reset = 'V = Vreset', method = 'euler', refractory = tau_ref)
	Neurons.V = 'rand()*(Vth-Vreset) + Vreset'
	Neurons.I = '60*mV * i / 999'

	SpikeMon = SpikeMonitor(Neurons, record = True)

	run(1000*ms, report = 'stdout')

	return Neurons.I / mV, np.array( SpikeMon.count )

def LIFe(taum = 20, tau_ref = 2.0, Vrest =  0, Vreset = 10, Vth = 20, R = 1):

	r'''
	  Neuron Parameters
	  tau_m : Membrane time constant
	  tau_ref : Refractory period
	  Vrest   : Resting potential
	  Vreset  : Reset potential
	  Vth     : Threshold
	  eqs     : Model equation
	  v_ext   : External input frequency
	'''
	tau_m    =  taum * ms
	tau_ref  =  tau_ref * ms
	Vrest    =  Vrest * mV
	Vreset   =  Vreset * mV
	Vth      =  Vth * mV

	C, k, V0 = 1.0, 1.79 * 1/mV, -35.50 * mV

	eqs = '''
		dV/dt = -(V - Vrest - R*I) / tau_m : volt (unless refractory)
		phi = C / (1 + exp(-k*(V-V0))) : 1
		I : volt
		'''

	Neurons = NeuronGroup(1000, model = eqs, threshold = 'rand()<phi', reset = 'V = Vreset', method = 'euler', refractory = tau_ref)
	Neurons.V = 'rand()*(Vth-Vreset) + Vreset'
	Neurons.I = '60*mV * i / 999'

	SpikeMon = SpikeMonitor(Neurons, record = True)

	run(1000*ms, report = 'stdout')

	return Neurons.I / mV, np.array( SpikeMon.count )

Id, Fd = LIF(taum = 20, tau_ref = 2.0, Vrest =  0, Vreset = 10, Vth = 20, R = 1)

tm = 20
tr = 2.0
vr = -73

Resistence = np.linspace(1,  10, 10)
tau        = np.linspace(15, 30, 10)
table      = []
for i in range(0, len(Resistence)):
	for j in range(0, len(tau)):
		print 'Resistence = ' + str(Resistence[i]) + ', tau = ' + str(tau[j])
		Is, Fs = LIFe(taum = tau[j], tau_ref = tr, Vrest =  vr, Vreset = vr+10, Vth = vr+20, R = Resistence[i])
		Q = np.mean( (Fs - Fd)**2 ) 
		table.append([Q, Resistence[i], tau[j]])	

table = np.array(table)
idx   = table[:, 0].argmin()

Is, Fs = LIFe(taum = table[idx, 2], tau_ref = tr, Vrest =  vr, Vreset = vr+10, Vth = vr+20, R = table[idx, 1])

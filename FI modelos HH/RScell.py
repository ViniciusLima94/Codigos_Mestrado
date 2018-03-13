# Neuron cell (RS)
from neuron import h
#import matplotlib.pyplot as plt
import numpy as np
import sys
reload(sys)
sys.setdefaultencoding('utf8')

h.load_file('stdrun.hoc')

h.celsius = 36
def simulate(I = 0.8, curr = True):

	soma = h.Section(name = 'soma')
	soma.L = soma.diam = 96
	soma.nseg = 1
	soma.Ra = 100
	soma.cm = 1

	soma.insert('pas')
	soma.e_pas = -70
	soma.g_pas = 0.0001
	soma.insert('hh2')
	soma.ek  = -100
	soma.ena = 50
	soma.gnabar_hh2 = 0.05
	soma.gkbar_hh2 = 0.005
	soma.insert('im')
	soma.gkbar_im = 7e-5
	soma.taumax_im = 1000

	apc = h.APCount(soma(0.5))
	v1 = h.Vector()
	apc.record(v1)

	h.v_init = -70
	
	stim = h.IClamp(soma(0.5))
	stim.delay = 0
	stim.dur = 1000
	stim.amp = I
	'''
	fl = h.Gfluct2(soma(0.5))
	fl.std_e = 0.012
	fl.std_i = 0.0264
	'''
	t_vec = h.Vector()
	v_vec = h.Vector()
	t_vec.record(h._ref_t)
	v_vec.record(soma(0.5)._ref_v)

	h.tstop = 1000
	h.run()

	return apc.n#, np.array(v1)



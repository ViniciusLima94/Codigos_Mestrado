# GL neuron IF curve

from brian2 import *
import sys
reload(sys)
sys.setdefaultencoding('utf8')


delta_t  = .1*ms
defaultclock.dt = delta_t
mu = .99
Vr = -65*mV        # Rest potential

tsim = 1*second

eqs = '''
	dV/dt =  (-V + Vr + R*I)/tau : volt (unless refractory)
	tau   : second
	R   : ohm
	A   : 1
	B  : volt
	C   : 1/volt
	phi   = 0*A / ( 1 + 10**(C*(B-V)) ) : 1
	I : amp
'''

n = NeuronGroup(num, eqs, threshold='rand() < phi', reset='V = Vr', method='linear', refractory=1.80*ms)
n.V = Vr
n.tau = delta_t / (1 - mu)
n.R   = 28.10 * Mohm
n.A = 0.99    * 1
n.B = -36.5753   * mV
n.C = 1.0720    * 1/mV
n.I = '5*nA* i / (num-1)'

monitor = SpikeMonitor(n)

run(tsim)
#plot(I2,f)
plot(n.I/nA, monitor.count / tsim)
xlabel(r'Corrente injetada [pA]', fontsize=15)
ylabel(r'Frequência de disparos [Hz]', fontsize=15)
savefig('curvafi.pdf', dpi = 600)

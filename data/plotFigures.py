import numpy as np 
import matplotlib.pyplot as plt 

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

vm = np.arange(-65.0, -35, 0.1)
delta_t = np.linspace(0.1, 1, 4)
tsim   = np.array([1e6])

phi_real = [phiv(x) for x in vm] 

plt.figure()
sub_index = 1
Q = []
for dt in delta_t:
	plt.subplot(2,2,sub_index)
	plt.plot(vm, phi_real, 'k')
	plt.title(r'$\Delta t = $'+ str(dt) +' [ms]')
	for T in tsim:
		filename = 'Tsim_' + str(int(T)) + '_dt_' + str(int(10*dt)) + '.csv'
		data = np.genfromtxt(filename, delimiter = ',')
		v = data[:,0]
		p = data[:,1]
		pst = data[:,2]
		preal = data[:,3]
		Q.append(1.0/len(preal) * np.sum( (pst-preal)**2 ))
		plt.plot(v, p, 'ro')

	sub_index += 1
plt.tight_layout()
plt.savefig('../figures/teste_metodo.pdf', dpi = 600)

# Q values
plt.figure()
plt.plot(delta_t, Q, 'o-')
plt.tight_layout()
plt.savefig('../figures/teste_metodo_q.pdf', dpi = 600)
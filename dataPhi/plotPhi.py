import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize

neuron_type = 'FS'

def fitPhi(neuron_type = 'FS'):
	if neuron_type == 'RS':
		RSphi = np.genfromtxt('RSphi.dat', delimiter = ',')
		xdata = RSphi[:,0]
		ydata = RSphi[:,1]
	else:
		FSphi = np.genfromtxt('FSphi.dat', delimiter = ',')
		xdata = FSphi[:,0]
		ydata = FSphi[:,1]

	#plt.figure()
	#plt.scatter(xdata, ydata)
	#plt.xlim([-40, -32])
	#plt.savefig(neuron_type + '_phi_exp.pdf', dpi = 600)

	def sigmoid(p,x):
	    x0,y0,c,k=p
	    y = 1.0 / (1 + np.exp(-k*(x-x0))) + y0
	    return y

	def residuals(p,x,y):
	    return y - sigmoid(p,x)

	def resize(arr,lower=0.0,upper=1.0):
	    arr=arr.copy()
	    if lower>upper: lower,upper=upper,lower
	    arr -= arr.min()
	    arr *= (upper-lower)/arr.max()
	    arr += lower
	    return arr

	# raw data
	x = xdata
	y = ydata

	p_guess=(np.median(x),np.median(y),1.0,1.0)
	p, cov, infodict, mesg, ier = scipy.optimize.leastsq(
	    residuals,p_guess,args=(x,y),full_output=1)  

	x0,y0,c,k=p
	print('''\
	x0 = {x0}
	y0 = {y0}
	c = {c}
	k = {k}
	'''.format(x0=x0,y0=y0,c=c,k=k))

	xp = np.linspace(-70, 0, 1500)
	pxp=sigmoid(p,xp)

	# Plot the results
	#plt.plot(x, y, '.', xp, pxp, '-')
	#plt.grid(True)
	return x, y, xp, pxp

v1, p1, vf1, vp1 = fitPhi('RS')
v2, p2, vf2, vp2 = fitPhi('FS')

plt.figure()
plt.plot(v1, p1, 'b.')
plt.plot(vf1, vp1, 'b-')
plt.plot(v2, p2, 'k.')
plt.plot(vf2, vp2, 'k-')
plt.xlim([-40, -30])
plt.legend([r'$\phi(v)_{RS}$', r'Sigmoid fit for $\phi(v)_{RS}$', r'$\phi(v)_{FS}$', r'Sigmoid fit for $\phi(v)_{FS}$'])
plt.savefig('phi_rs_fs_fit.pdf', dpi=600)

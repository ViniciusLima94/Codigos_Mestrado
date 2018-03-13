'''
	F-I curves for RS and FS cortical cells with minimal H&H model.
'''
import RScell as rs
import FScell as fs
import matplotlib.pyplot as plt
import numpy as np

I = np.arange(0, 5, 0.1)  # Current in nA
Frs = []
Ffs = []

[Frs.append(rs.simulate(curr)) for curr in I]
[Ffs.append(fs.simulate(curr)) for curr in I]

'''
plt.plot(I, Frs)
plt.plot(I, Ffs)
plt.ylabel('Frequency [Hz]')
plt.xlabel('Current [nA]')
plt.legend(['RS', 'FS'], loc = 2)
plt.show()
'''

np.savetxt('fi_modelohh.dat', np.array([Frs, Ffs]).T, delimiter = ',' )


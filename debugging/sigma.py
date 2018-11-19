import numpy as np 
a = np.fromfile('out_sigma_ref.txt',sep=',')
b = np.fromfile('out_sigma_simulated.txt',sep=',')
print np.std(b-a)
print np.mean(b-a)

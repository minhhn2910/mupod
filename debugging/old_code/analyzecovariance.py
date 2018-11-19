import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
import math
n4_ref = np.fromfile("n4_ref.txt",sep=",")
n4_err = np.fromfile("n4_err.txt",sep=",")

#print ('n6 std %f '%(np.std(n6_ref-n6_err)))
c = n4_ref-n4_err
#c = np.random.uniform(-1,1,8000000)
print len(c)
axis2 = 500
d = c.reshape([10000,500])
d = d.T
n4_std = []
for i in range(axis2):
    n4_std.append(np.std(d[i]))

cov_n4 = []

for i in range(axis2):
    for j in range(i+1,axis2):
        temp_d1 = d[i]
        temp_d2 = d[j]
        temp_cov = temp_d1.dot(temp_d2)/len(temp_d1)
        cov_n4.append(temp_cov)

print cov_n4
plt.hist(cov_n4)
plt.show()
n4_std_sqr = map(lambda x: x**2, n4_std)

print 2*sum(cov_n4)
print sum(n4_std_sqr)
print (math.sqrt(sum(n4_std_sqr)+2*sum(cov_n4)))
'''
c = n3_ref-n3_err
d = c.reshape([10000,10])
d = d.T
n3_std = []
for i in range(10):
    n3_std.append(np.std(d[i]))
print n4_std[:5]
print n3_std[:5]
'''
plt.hist(cov_n4, bins=100)
plt.show()

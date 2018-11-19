import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats

n4_ref = np.fromfile("n4_ref.txt",sep=",")
n4_err = np.fromfile("n4_err.txt",sep=",")
n3_ref = np.fromfile("n3_ref.txt",sep=",")
n3_err = np.fromfile("n3_err.txt",sep=",")
n6_ref = np.fromfile("n6_ref.txt",sep=",")
n6_err = np.fromfile("n6_err.txt",sep=",")

out6_ref = np.fromfile("out6_ref.txt",sep=",")
out6_err = np.fromfile("out6_err.txt",sep=",")

print ('n3 std %f '%(np.std(n3_ref-n3_err)))
print ('n4 std %f '%(np.std(n4_ref-n4_err)))
print ('n6 std %f '%(np.std(n6_ref-n6_err)))
print ('out6 std %f '%(np.std(out6_ref-out6_err)))

'''
c = n4_ref-n4_err
print len(c)
d = c.reshape([10000,500])
d = d.T
n4_std = []
for i in range(500):
    n4_std.append(np.std(d[i]))
c = n3_ref-n3_err
d = c.reshape([10000,10])
d = d.T
n3_std = []
for i in range(10):
    n3_std.append(np.std(d[i]))

c = n5_ref-n5_err
d = c.reshape([10000,800])
d = d.T
n5_std = []
for i in range(10):
    n5_std.append(np.std(d[i]))


print 'n4_std'
print sum(n4_std)/len(n4_std)
print 'n3_std'
print sum(n3_std)/len(n3_std)

print 'n5_std'
print sum(n5_std)/len(n5_std)

#plt.hist(n4_ref-n4_err, bins=100)
#plt.show()
'''

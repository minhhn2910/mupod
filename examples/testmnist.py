import os
import caffe
os.environ['GLOG_minloglevel'] = '2' 
import numpy
from pylab import *
zeros
import os
solver = None  # ignore this workaround for lmdb data (can't instantiate two solvers on the same data)
solver_config_path = 'test_mnist/custom_auto_solver.prototxt'
solver = caffe.get_solver(solver_config_path)
niter = 250  # EDIT HERE increase to train for longer
test_interval = niter / 10
# losses will also be stored in the log
train_loss = zeros(niter)
test_acc = zeros(int(np.ceil(niter / test_interval)))
train_loss = zeros(niter)
for it in range(niter):
    solver.step(1)  # SGD by Caffe
    
    # store the train loss
    train_loss[it] = solver.net.blobs['loss'].data
    
    # run a full test every so often
    # (Caffe can also do this for us and write to a log, but we show here
    #  how to do it directly in Python, where more complicated things are easier.)
    if it % test_interval == 0:
        print 'Iteration', it, 'testing...'
        correct = 0
        for test_it in range(100):
            solver.test_nets[0].forward()
            correct += sum(solver.test_nets[0].blobs['score'].data.argmax(1)
                           == solver.test_nets[0].blobs['label'].data)
        test_acc[it // test_interval] = correct / 1e4
print test_acc
import readline
readline.write_history_file('testpython')

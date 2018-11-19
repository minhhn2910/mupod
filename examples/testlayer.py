import caffe
import numpy as np

class CustomActivationLayer(caffe.Layer):

	def setup(self, bottom, top):
        # check input pair
        #if len(bottom) != 2:
        #    raise Exception("Need two inputs to compute distance.")
		pass

	def reshape(self, bottom, top):
		#pass
        # check input dimensions match
        #if bottom[0].count != bottom[1].count:
        #    raise Exception("Inputs must have the same dimension.")
        # difference is shape of inputs
        #self.diff = np.zeros_like(bottom[0].data, dtype=np.float32)
        # loss output is scalar
        #top[0].reshape(1)
		top[0].reshape(*bottom[0].data.shape)
	def clip_value(x):
		if (x>0 and x<1):
			return x
		else:
			return 0
	def forward(self, bottom, top):
		#~ print "Test "
		#count = bottom[0].count
		#~ print count
		#~ print top[0].count
		#~ top[0].data[...] = map(lambda x: 5 if x>5 else x*(x>0), bottom[0].data)
		top[0].data[...] = map(lambda x: (x<10)*x*(x>0) + (x>=10)*1, bottom[0].data)
		#top[0].data[...] = bottom[0].data[...]
		#for i in range(count-1):
		#	if(bottom[0].data[i] < 0):
		#		top[0].data[i] = 0
		#	else:
		#		top[0].data[i] = bottom[0].data[i]

	def backward(self, top, propagate_down, bottom):
		#count = bottom[0].count
		bottom[0].diff[...] = top[0].diff[...] * (bottom[0].data[...] >0)
		#for i in range(count):
		#	bottom[0].diff[i] = top[0].diff[i] * (bottom[0].data[i] >0)

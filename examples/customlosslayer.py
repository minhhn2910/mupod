import caffe
import numpy as np
import pickle
import glog
class SoftmaxLossModLayer(caffe.Layer):
    """
    Compute the Softmax Loss in the same manner but consider soft labels
    as the ground truth
    """
    batch_size = 1
    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 2:
            raise Exception("Need two inputs to compute distance.")

    def reshape(self, bottom, top):
        # check input dimensions match
        if bottom[0].num != bottom[1].num:
            self.batch_size = max(bottom[0].count,bottom[1].count)/min(bottom[0].count,bottom[1].count)
            #raise Exception("Inputs must have the same dimension.")
        # difference is shape of inputs
        self.diff = np.zeros_like(bottom[0].data, dtype=np.float32)
        # loss output is scalar
        top[0].reshape(1)

    def forward(self, bottom, top):        
       # sim_mat = np.load('...')#path to label mapping
        
        scores = bottom[0].data
        gt_score = np.zeros((bottom[0].num,10))
        if bottom[0].num != 1:
            list_data = list(bottom[1].data)
            for i in range(bottom[0].num):
                gt_score[i,int(list_data[i])] = 1
        #gt_score[range(bottom[0].num),:] = sim_mat[,:]        
        #normalizing to avoid instability
        #pickle.dump({'pd': bottom[0].data, 'gt': bottom[1].data, 'encoded' : gt_score}, open('/home/minh/data_dump.pkl','w'))
        scores -= np.max(scores)
        exp_scores = np.exp(scores)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        #glog.info("%s "%(str(probs))) 
        logprob = -np.log(probs)
        
        data_loss = np.sum(np.sum(gt_score*logprob,axis=1))/bottom[0].num
       # pickle.dump({'exp': exp_scores, 'prob': probs, 'loss' :np.sum(gt_score*logprob,axis=1)}, open('/home/minh/score_dump.pkl','w'))
        glog.info('%f %d' % (data_loss, bottom[0].num))
        self.diff[...] = probs
        top[0].data[...] = data_loss

    def backward(self, top, propagate_down, bottom):
        delta = self.diff
        glog.info('in backward prob')
        #sim_mat = np.load('...')#path to label mapping
        gt_score = np.zeros((bottom[0].num,10))
        list_data = list(bottom[1].data)
        for i in range(bottom[0].num):
            gt_score[i,int(list_data[i])] = 1
        #gt_score[range(bottom[0].num),:] = sim_mat[list(bottom[1].data),:] 
        for i in range(2):
            if not propagate_down[i]:
                continue
            if i==0:
                delta[range(bottom[0].num), np.array(bottom[1].data,dtype=np.uint16)] -= 1
                delta = delta*gt_score
            bottom[i].diff[...] = delta/bottom[0].num
            glog.info('%s'%(str(bottom[i].diff)))
        
        
        #pickle.dump({'delta': delta, 'diff_0': bottom[0].diff,  'diff_1': bottom[1].diff}, open('/home/minh/delta_dump.pkl','w'))
    

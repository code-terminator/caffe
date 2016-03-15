
# coding: utf-8

# In[7]:

"""
This function is used train the PAT model
"""
import sys
sys.path.append("/home/bugfree/Workspace/caffe/python")
import caffe
import numpy as np
import os
import time
import cPickle
from caffe.proto.caffe_pb2 import SolverParameter
from pat_model import pat_net
sys.path.append("/home/bugfree/Workspace/caffe/python_ext")
from train import train_net



# In[2]:

class data_provider(object):
    """
    This class defines the data provider for training
    input:
        fname: the file name for training file.
        fcpickle: the existing cPickle file, contatins data and index.
            This is used to ensure, the learning can be reproduced.
        save_cpickle: indicate whether to store the generated data and label
            for future usage.
        batch_size: the batch size, default = 128
    """
    def __init__(self,fname,fcpickle='none',save_cpickle=0, batch_size=128):
        '''
        Class construction
        '''
        self.fname = fname
        self.fcpickle = fcpickle
        self.save_cpickle = save_cpickle
        self.batch_size = batch_size

    def load(self):
        '''
        This function is used only once for loading data into memory
        (assuming the data is not too large to store in RAM).
        '''
        if self.fcpickle != "none":
            [data,label] = cPickle.load(open(self.fcpickle,'rb'))
            speech_dim = data.shape[1]
            latentvar_dim = label.shape[1]
            num_sample = data.shape[0]
        else:
            f = open(self.fname,'r')
            data = list([])
            label = list([])

            # read the training speech (data) and latent variable (label) from the file
            for idx,line in enumerate(f):
                line = line.strip()
                speech, latentvar = line.split(';')
                speech = map(float, speech.strip().split(','))
                latentvar = map(float, latentvar.strip().split(','))
                # check the dimensionality
                if idx == 0:
                    speech_dim = len(speech)
                    latentvar_dim = len(latentvar)
                else:
                    if len(speech) != speech_dim or len(latentvar) != latentvar_dim:
                        raise ValueError('Dimensionality mismatched')
                data.append(speech)
                label.append(latentvar)

            # convert data and label into a ndarray
            data = np.array(data, dtype=np.float32)
            label = np.array(label, dtype=np.float32)
            num_sample = label.shape[0]

            # random shuffle
            idx = np.random.permutation(num_sample)
            data = data[idx,:]
            label = label[idx,:]

            # convert them into caffe required format
            data = data.reshape((num_sample,speech_dim))
            label = label.reshape((num_sample,latentvar_dim))

            # save as the cpickle file for future use if necessary
            if self.save_cpickle == 1:
                fcpickle = self.fname.strip().split('.')[0] + '.cpickle'
                cPickle.dump([data,label],open(fcpickle,'wb'))

        # store the input: data, label, data_dim, label_dim, num_sample
        self.data = data
        self.label = label
        self.data_dim = speech_dim
        self.label_dim = latentvar_dim
        self.num_sample = num_sample
        # also needs a variable: cur_idx used as a pointer
        self.cur_idx = 0

    def get_next_batch(self):
        '''
        This function is used to obtain the data for the next batch
        return:
            a dictionary with two keys: 'data' and 'label'
        '''
        # check the whether the next batch will exceed the num_sample constraints
        if self.cur_idx + self.batch_size > self.num_sample:
            data = self.data[self.cur_idx:self.num_sample,...]
            label = self.label[self.cur_idx:self.num_sample,...]
            self.cur_idx = 0
        else:
            end_idx = self.cur_idx + self.batch_size
            data = self.data[self.cur_idx:end_idx,...]
            label = self.label[self.cur_idx:end_idx,...]
            self.cur_idx = end_idx
        return {'data':data, 'label':label}


# In[3]:

def get_sgd_solver(test_iter, test_interval,snapshot_prefix,\
                   lr_policy="step", base_lr=0.01, momentum=0.9, gamma=0.1,\
                   snapshot=200000, display=100, stepsize=100000, maxiter=350000):
    '''
    This function is used to construct a solver for caffe deep model.
    It is equivlent as writing a solver_prototext
    Inputs:
        test_iter: specifies how many forward passes the test should carry out.
            For instance, if test_iter=100, and test_batch_size=100, then totally
            We will test 10,000 samples.
        test_interval: carry out testing very $test_interval training iterations
        base_lr: the base learning rate
        momentum: the momentum for sgd
        weight_decay: the weight decay for the network
        lr_policy: the learning rate policy
        gamma: for step learning plicy: drop the learning rate by a factor of 1/gamma.
        display: display every $display iterations
        max_iter: the maximum number of iterations
        snapshot: snapshot intermediate results, default true
        snapshot_prefix: the prefix for the snapshot file name
    Notes:
        There are also associated parameters you need to speicfy if you use other settings.
        For instance, you need to specify the stepsize, if your learning policy is "step".
        Detailed info please refer to the "SolverParameter" section in the file:
            caffe.proto
    '''
    return SolverParameter(test_iter=test_iter, test_interval=test_interval,\
                                snapshot_prefix=snapshot_prefix, lr_policy=lr_policy,\
                                base_lr=base_lr, momentum=momentum, gamma=gamma,\
                                snapshot=snapshot, display=display, stepsize=stepsize,\
                                                           max_iter=maxiter)


# In[4]:

def run_pat():
    #################################
    # variable you want to set
    #################################
    # set the path for the log file
    basedir = "/mnt/disk1/bugfree/PAT"
    basename = "pat1m-sgd"
    date = time.strftime("%d-%m-%Y")
    basename += "-" + date + "-"
    # fname_train = '/home/bugfree/Workspace/caffe/python_ext/data/pat/pat_1m_train.txt'
    # fname_test = '/home/bugfree/Workspace/caffe/python_ext/data/pat/pat_1w_test.txt'
    # fcpickle_train = '/home/bugfree/Workspace/caffe/python_ext/data/pat/pat_1m_train.cpickle'
    # fcpickle_test = '/home/bugfree/Workspace/caffe/python_ext/data/pat/pat_1w_test.cpickle'

    # fname_train = basedir + "/data/train_data_orig_scale.txt"
    # fname_test = basedir + "/data/test_data_orig_scale.txt"
    fcpickle_train = basedir + "/data/train_data_orig_scale.cpickle"
    fcpickle_test = basedir + "/data/test_data_orig_scale.cpickle"
    input_dim = 300
    output_dim = 1
    train_batch_size = 128
    test_batch_size = 128
    test_iter = [100]
    test_interval = 100000
    lr_policy = "step"
    lr = 0.001
    gamma = 0.1
    snapshot = 250000
    stepsize = 5000000
    maxiter = 10000000


    # initialize the log file
    caffe.init_glog()

    if not os.path.exists(basedir):
        os.mkdir(basedir)


    # train_provider = data_provider(fname_train,save_cpickle=1, batch_size=train_batch_size)
    # test_provider = data_provider(fname_test,save_cpickle=1, batch_size=test_batch_size)
    train_provider = data_provider('',fcpickle=fcpickle_train, batch_size=train_batch_size)
    test_provider = data_provider('',fcpickle=fcpickle_test, batch_size=test_batch_size)
    # net = pat_net(300,33,train_batch_size,test_batch_size)
    net = pat_net(input_dim,output_dim,train_batch_size,test_batch_size)
    expname = basename + "%g" % (lr)
    out_dir = "%s/%s/" %(basedir, expname)
    solver = get_sgd_solver(test_iter=test_iter, test_interval=test_interval,\
                            snapshot_prefix=out_dir, lr_policy="step", \
                            base_lr=lr, stepsize=stepsize, maxiter=maxiter,\
                            snapshot=snapshot, gamma=gamma)
    train_net(solver=solver, net=net, data_provider=[train_provider,test_provider],\
              output_dir=out_dir, maxiter=maxiter,log_to_file=True)


# In[5]:

if __name__ == '__main__':
    run_pat()

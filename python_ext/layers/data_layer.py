
# coding: utf-8

# In[12]:

import sys
sys.path.append("/home/bugfree/Workspace/caffe/python")
import caffe
from multiprocessing import Process, Queue
import yaml
import numpy as np


# In[13]:

class BlobFetcher(Process):
    def __init__(self, queue, data_provider):
        '''
        Initilizing a blob fetcher
        Note:
            data_provier: either for training or testing
        '''
        super(BlobFetcher, self).__init__()
        self.name = "BlobFetcher"
        self._queue = queue
        self._data_provider = data_provider

    def run(self):
        '''
        This function defines what each fetcher does
        '''
        print "BlobFetcher Started"
        self._data_provider.load()
        # load the next batch and put it into the queue
        while True:
            blobs = self._data_provider.get_next_batch()
            self._queue.put(blobs)



# In[14]:

class DataLayer(caffe.Layer):
    """
    General Data Layer for batch of inputs.
    """
    def set_data(self, data_provider):
        '''
        This function is called at the begining of the training to set
        a length of 10 queue, so that each queue is fetching to load a batch
        of data.  This will increase the speed since there is no need to wait
        the data loading process everytime.
        Input:
            data_provider: either the provider for training or testing.
        '''
        self._blob_queue = Queue(10)
        # here, the data_provider is either the training_provider
        # or the testing provider.  It is not the set of both of them
        self._prefetch_process = BlobFetcher(self._blob_queue, data_provider)
        # start the fetching, which will excute the "run" function of BlobFetcher
        self._prefetch_process.start()

        def cleanup():
            '''
            This is termination function
            '''
            self._prefetch_process.terminate()
            self._prefetch_process.join()

        import atexit
        atexit.register(cleanup)


    def setup(self, bottom, top):
        '''
        This function is requried by caffe, which is used to determine the
        shape of the blob.  The parse layer parameter string must be a valid YAML.
        Input:
            bottom: null
            top: the data and label
        '''
        # obtain the param_str, this is parsed when we construct such a python layer
        param_dict = yaml.load(self.param_str)
        # obtain the name of the top layer
        self.top_names = param_dict['top_names']
        # obtain the shape of the top
        self.top_shapes = param_dict['top_shapes']
        # reshape the data layer into the same shape as the input described
        self._name_to_top_map = {}
        for idx, top_name in enumerate(self.top_names):
            top[idx].reshape(*self.top_shapes[idx])
            self._name_to_top_map[top_name] = idx

    def forward(self, bottom, top):
        '''
        This is a forward function, which is required to define by caffe.
        This funtion is used to calculate the forward/function value from bottom
        to top.  Besically, we need to get blobs and copy them into this layer's
        top blob vector
        Input:
            bottom: null
            top: the data and label as blobs
        '''
        blobs = self._blob_queue.get()

        for blob_name, blob in blobs.iteritems():
            # obtain the index by searching the blob's name
            top_idx = self._name_to_top_map[blob_name]
            # reshape net's input blobs
            top[top_idx].reshape(*(blob.shape))
            # copy data into net's input blobs
            top[top_idx].data[...] = blob.astype(np.float32, copy=False)

    def backward(self, bottom, top, propagate_down):
        '''
        This is a backward function, which is required to define by caffe.
        However, as a data layer, there is no bottom
        '''
        pass

    def reshape(self, bottom, top):
        '''
        Reshaping happens during the call to forward.
        '''
        pass


# In[11]:

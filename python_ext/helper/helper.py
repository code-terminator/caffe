import caffe
import numpy as np
import matplotlib.pyplot as plt
import cPickle

if __name__ == "__main__":
    # the variable for either using a GPU (set to 1) to compute
    # or a CPU (set to 0)
    GPU = 1;
    proto_root = "/mnt/disk1/bugfree/PAT/pat1m-sgd0.01"
    # NOTE: this prototext have to be a deploy version of the prototxt
    # It is not the one, we used for training
    net_deploy = proto_root + "/net.deploy"
    model = proto_root + "/_iter_7500000.caffemodel"
    data_root = "/home/bugfree/Workspace/caffe/python_ext/data/pat"
    fdata = data_root + "/pat_1w_test.cpickle"
    data_layer_name = 'Python1'

    # set the GPU/CPU mode
    if GPU == 1:
        caffe.set_device(0)
        caffe.set_mode_gpu()
    else:
        caffe.set_mode_cpu()

    # load the net from net prototext
    # in the same time load the caffemodel (it contains the model parameters)
    net = caffe.Net(net_deploy,model,caffe.TEST)

    # load data
    [data,label] = cPickle.load(open(fdata,'rb'))
    print data.shape

    # configure preprocessing
    input_shape = list(net.blobs[data_layer_name].data.shape)
    # print input_shape
    # reshape the data layer on the fly
    net.blobs[data_layer_name].reshape(data.shape[0],data.shape[1])
    # load the data to the caffe net
    net.blobs[data_layer_name].data[...] = data
    # compute
    out = net.forward()
    prediction = out['InnerProduct4']

    # for i in range(prediction.shape[0]):
    for i in range(5):
        ground = label[i,:]
        pred = prediction[i,:]
        print ground
        print pred
        print "======================="

    # check the performance for each dimension
    print sum((label - prediction)**2,0)/prediction.shape[0]
    print sum(sum((label - prediction)**2,0)/prediction.shape[0])



    # print the blob's name and the shape of data
    # print [(k, v.data.shape) for k, v in net.blobs.items()]
    # print the weight and bias terms for each layers that contains parameter
    # print [(k, v[0].data.shape, v[1].data.shape) for k, v in net.params.items()]

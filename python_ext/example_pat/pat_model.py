import sys
sys.path.append("/home/bugfree/Workspace/caffe/python")
from caffe import layers as L, params as P, to_proto
import yaml
from IPython.core.debugger import Tracer;

def full_sig(bottom,num_output,weight_filler={'type':'gaussian','std':0.01},\
            bias_term=True, bias_filler={'type':'constant','value':0}):
    '''
    This function constructs the fully connect - sigmod pair
    Inputs:
        bottom: the bottom layer connection
        num_output: the number of filters
        weight_filler: this is used for initializing the weights
        bias_term: whether you need bias term for each filter
        bias_filler: used for initializing the bias
    '''
    fully = L.InnerProduct(bottom, num_output=num_output,\
                               weight_filler=weight_filler,\
                               bias_term=bias_term, bias_filler=bias_filler)
    sig = L.Sigmoid(fully)
    return fully, sig

def full_relu(bottom,num_output,weight_filler={'type':'gaussian','std':0.01},\
            bias_term=True, bias_filler={'type':'constant','value':0}):
    fully = L.InnerProduct(bottom, num_output=num_output,\
                               weight_filler=weight_filler,\
                               bias_term=bias_term, bias_filler=bias_filler)
    relu = L.ReLU(fully)
    return fully, relu


def make_python_data_layer(top_names,top_shapes):
    '''
    This function is used to make the data layer from python
    Inputs:
        top_names: the name of the top blobs, in the form of list,
            the length is equal to the number of top blobs
        top_shapes: the shape of the top blobs
    Returns:
        A python layer
    '''
    param_dict = {'top_names':top_names, 'top_shapes':top_shapes}
    # serialization to string
    yaml_str = yaml.dump(param_dict)
    # ntop: number of top blobs
    return L.Python(module="python_ext",layer="DataLayer",param_str=yaml_str,\
                        ntop=len(top_names))

def pat_net(data_dim,label_dim,train_batch_size=128,test_batch_size=128):
    '''
    This funtion defines the connectivity of our PAT deep network
    Inputs:
        batach_size
    '''
    # the top dimension for each fully connected layer
    fully1_dim = data_dim
    fully2_dim = data_dim
    fully3_dim = data_dim

    data,label = make_python_data_layer(["data","label"],[(train_batch_size,data_dim),(test_batch_size,label_dim)])
    fully1, sig1 = full_sig(data,fully1_dim)
    fully2, sig2 = full_sig(sig1,fully1_dim)
    fully3, sig3 = full_sig(sig2,fully1_dim)
    fully4 = L.InnerProduct(sig3,num_output=label_dim,bias_term=True,\
                                weight_filler={'type':'gaussian','std':0.01},\
                                bias_filler={'type':'constant','value':0})
    # define the L2 regression loss
    loss = L.EuclideanLoss(fully4, label)
    return to_proto(loss)

def pat_net_relu(data_dim,label_dim,train_batch_size=128,test_batch_size=128):
    '''
    This funtion defines the connectivity of our PAT deep network
    Inputs:
        batach_size
    '''
    # the top dimension for each fully connected layer
    fully1_dim = data_dim
    fully2_dim = data_dim
    fully3_dim = data_dim

    data,label = make_python_data_layer(["data","label"],[(train_batch_size,data_dim),(test_batch_size,label_dim)])
    fully1, relu1 = full_relu(data,fully1_dim)
    fully2, relu2 = full_relu(relu1,fully2_dim)
    fully3, relu3 = full_relu(relu2,fully3_dim)
    fully4 = L.InnerProduct(relu3,num_output=label_dim,bias_term=True,\
                                weight_filler={'type':'gaussian','std':0.01},\
                                bias_filler={'type':'constant','value':0})
    # define the L2 regression loss
    loss = L.EuclideanLoss(fully4, label)
    return to_proto(loss)

def pat_net_one_relu(data_dim,label_dim,train_batch_size=128,test_batch_size=128):
    '''
    Construct one 300x300 layer followed by a relu, and them linear embedding
    '''
    # the top dimension for each fully connected layer
    fully1_dim = 500

    data,label = make_python_data_layer(["data","label"],[(train_batch_size,data_dim),(test_batch_size,label_dim)])
    fully1, relu1 = full_relu(data,fully1_dim)
    fully2 = L.InnerProduct(relu1,num_output=label_dim,bias_term=True,\
                                weight_filler={'type':'gaussian','std':0.01},\
                                bias_filler={'type':'constant','value':0})
    loss = L.EuclideanLoss(fully2, label)
    return to_proto(loss)

if __name__ == '__main__':
    debug_here = Tracer()
    pat_net(data_dim=300,label_dim=30)

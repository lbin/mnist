import numpy as np
import pandas as pd
import mxnet as mx
import logging

# set to 0 to train on all available data
VALIDATION_SIZE = 2000

# create the training
dataset = pd.read_csv("./train.csv")
target = dataset[[0]].values.ravel()
train = dataset.iloc[:,1:].values

def conv_factory(data, num_filter, kernel, stride, pad, act_type = 'relu', conv_type = 0):
    if conv_type == 0:
        conv = mx.symbol.Convolution(data = data, num_filter = num_filter, kernel = kernel, stride = stride, pad = pad)
        bn = mx.symbol.BatchNorm(data=conv)
        act = mx.symbol.Activation(data = bn, act_type=act_type)
        return act
    elif conv_type == 1:
        conv = mx.symbol.Convolution(data = data, num_filter = num_filter, kernel = kernel, stride = stride, pad = pad)
        bn = mx.symbol.BatchNorm(data=conv)
        return bn

def residual_factory(data, num_filter, dim_match):
    if dim_match == True: # if dimension match
        identity_data = data
        conv1 = conv_factory(data=data, num_filter=num_filter, kernel=(3,3), stride=(1,1), pad=(1,1), act_type='relu', conv_type=0)

        conv2 = conv_factory(data=conv1, num_filter=num_filter, kernel=(3,3), stride=(1,1), pad=(1,1), conv_type=1)
        new_data = identity_data + conv2
        act = mx.symbol.Activation(data=new_data, act_type='relu')
        return act
    else:
        conv1 = conv_factory(data=data, num_filter=num_filter, kernel=(3,3), stride=(2,2), pad=(1,1), act_type='relu', conv_type=0)
        conv2 = conv_factory(data=conv1, num_filter=num_filter, kernel=(3,3), stride=(1,1), pad=(1,1), conv_type=1)

        # adopt project method in the paper when dimension increased
        project_data = conv_factory(data=data, num_filter=num_filter, kernel=(1,1), stride=(2,2), pad=(0,0), conv_type=1)
        new_data = project_data + conv2
        act = mx.symbol.Activation(data=new_data, act_type='relu')
        return act

def residual_net(data, n):
    #fisrt 2n layers
    for i in range(n):
        data = residual_factory(data=data, num_filter=16, dim_match=True)

    #second 2n layers
    for i in range(n):
        if i==0:
            data = residual_factory(data=data, num_filter=32, dim_match=False)
        else:
            data = residual_factory(data=data, num_filter=32, dim_match=True)

    #third 2n layers
    for i in range(n):
        if i==0:
            data = residual_factory(data=data, num_filter=64, dim_match=False)
        else:
            data = residual_factory(data=data, num_filter=64, dim_match=True)

    return data

def get_symbol(num_classes = 10):
    conv = conv_factory(data=mx.symbol.Variable(name='data'), num_filter=16, kernel=(3,3), stride=(1,1), pad=(1,1), act_type='relu', conv_type=0)
    n = 9 # set n = 3 means get a model with 3*6+2=20 layers, set n = 9 means 9*6+2=56 layers
    resnet = residual_net(conv, n) #
    pool = mx.symbol.Pooling(data=resnet, kernel=(7,7), pool_type='avg')
    flatten = mx.symbol.Flatten(data=pool, name='flatten')
    fc = mx.symbol.FullyConnected(data=flatten, num_hidden=num_classes,  name='fc1')
    softmax = mx.symbol.SoftmaxOutput(data=fc, name='softmax')
    return softmax

# split dataset
val_data = train[:VALIDATION_SIZE].astype('float32')
val_label = target[:VALIDATION_SIZE]
train_data = train[VALIDATION_SIZE: , :].astype('float32')
train_label = target[VALIDATION_SIZE:]
train_data = np.array(train_data).reshape((-1, 1, 28, 28))
val_data = np.array(val_data).reshape((-1, 1, 28, 28))

# Normalize data
train_data[:] /= 256.0
val_data[:]/= 256.0

batch_size = 500
train_iter = mx.io.NDArrayIter(train_data, train_label, batch_size=batch_size, shuffle=True)
val_iter = mx.io.NDArrayIter(val_data, val_label, batch_size=batch_size)

# logging
head = '%(asctime)-15s Node[0] %(message)s'
logging.basicConfig(level=logging.DEBUG, format=head)

model_args = {}
num_epoch = 200
tmp = mx.model.FeedForward.load('resnet', 100)
model_args = {'arg_params' : tmp.arg_params,
              'aux_params' : tmp.aux_params,
              'begin_epoch' : 100}

# create model
devs = mx.gpu(0)
network=get_symbol()
model = mx.model.FeedForward(
        ctx                = devs,
        symbol             = network,
        num_epoch          = num_epoch,
        learning_rate      = 0.01,
        momentum           = 0.9,
        wd                 = 0.0001,
        initializer        = mx.init.Xavier(factor_type="in", magnitude=2.34),
        **model_args)

eval_metrics = ['accuracy']
model.fit(
	X=train_iter,
	eval_metric        = eval_metrics,
	eval_data	 = val_iter
	)

prefix = 'resnet'
model.save(prefix, num_epoch)

#predict
test = pd.read_csv("./test.csv").values
test_data = test.astype('float32')
test_data = np.array(test_data).reshape((-1, 1, 28, 28))
test_data[:]/= 256.0
test_iter = mx.io.NDArrayIter(test_data, batch_size=batch_size)

pred = model.predict(X = test_iter)
pred = np.argsort(pred)
np.savetxt('submission_lb_res_v1_1.csv', np.c_[range(1,len(test)+1),pred[:,9]], delimiter=',', header = 'ImageId,Label', comments = '', fmt='%d')

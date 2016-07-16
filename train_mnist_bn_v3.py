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


# Basic Conv + BN + ReLU factory
def ConvFactory(data, num_filter, kernel, stride=(1,1), pad=(0, 0), act_type="relu", mirror_attr={}):
    conv = mx.symbol.Convolution(data=data, num_filter=num_filter, kernel=kernel, stride=stride, pad=pad)
    bn = mx.symbol.BatchNorm(data=conv)
    act = mx.symbol.Activation(data = bn, act_type=act_type, attr=mirror_attr)
    return act

# A Simple Downsampling Factory
def DownsampleFactory(data, ch_3x3, mirror_attr):
    # conv 3x3
    conv = ConvFactory(data=data, kernel=(3, 3), stride=(2, 2), num_filter=ch_3x3, pad=(1, 1), mirror_attr=mirror_attr)
    # pool
    pool = mx.symbol.Pooling(data=data, kernel=(3, 3), stride=(2, 2), pad=(1, 1), pool_type='max', attr=mirror_attr)
    # concat
    concat = mx.symbol.Concat(*[conv, pool])
    return concat

# A Simple module
def SimpleFactory(data, ch_1x1, ch_3x3, mirror_attr):
    # 1x1
    conv1x1 = ConvFactory(data=data, kernel=(1, 1), pad=(0, 0), num_filter=ch_1x1, mirror_attr=mirror_attr)
    # 3x3
    conv3x3 = ConvFactory(data=data, kernel=(3, 3), pad=(1, 1), num_filter=ch_3x3, mirror_attr=mirror_attr)
    #concat
    concat = mx.symbol.Concat(*[conv1x1, conv3x3])
    return concat

def get_loc(data, attr={'lr_mult':'0.01'}):
    """
    the localisation network in lenet-stn, it will increase acc about more than 1%,
    when num-epoch >=15
    """
    loc = mx.symbol.Convolution(data=data, num_filter=30, kernel=(5, 5), stride=(2,2))
    loc = mx.symbol.Activation(data = loc, act_type='relu')
    loc = mx.symbol.Pooling(data=loc, kernel=(2, 2), stride=(2, 2), pool_type='max')
    loc = mx.symbol.Convolution(data=loc, num_filter=60, kernel=(3, 3), stride=(1,1), pad=(1, 1))
    loc = mx.symbol.Activation(data = loc, act_type='relu')
    loc = mx.symbol.Pooling(data=loc, global_pool=True, kernel=(2, 2), pool_type='avg')
    loc = mx.symbol.Flatten(data=loc)
    loc = mx.symbol.FullyConnected(data=loc, num_hidden=6, name="stn_loc", attr=attr)
    return loc

def get_symbol(num_classes = 10, force_mirroring=False):
    if force_mirroring:
        attr = {'force_mirroring': 'true'}
    else:
        attr = {}

    data = mx.symbol.Variable(name="data")
    data = mx.sym.SpatialTransformer(data=data, loc=get_loc(data), target_shape = (28,28),transform_type="affine", sampler_type="bilinear")
    conv1 = ConvFactory(data=data, kernel=(3,3), pad=(1,1), num_filter=96, act_type="relu", mirror_attr=attr)
    in3a = SimpleFactory(conv1, 32, 32, mirror_attr=attr)
    in3b = SimpleFactory(in3a, 32, 48, mirror_attr=attr)
    in3c = DownsampleFactory(in3b, 80, mirror_attr=attr)
    in4a = SimpleFactory(in3c, 112, 48, mirror_attr=attr)
    in4b = SimpleFactory(in4a, 96, 64, mirror_attr=attr)
    in4c = SimpleFactory(in4b, 80, 80, mirror_attr=attr)
    in4d = SimpleFactory(in4c, 48, 96, mirror_attr=attr)
    in4e = DownsampleFactory(in4d, 96, mirror_attr=attr)
    in5a = SimpleFactory(in4e, 176, 160, mirror_attr=attr)
    in5b = SimpleFactory(in5a, 176, 160, mirror_attr=attr)
    pool = mx.symbol.Pooling(data=in5b, pool_type="avg", kernel=(7,7), name="global_pool", attr=attr)
    flatten = mx.symbol.Flatten(data=pool,attr=attr)
    fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=500)
    tanh= mx.symbol.Activation(data=fc1, act_type="tanh")
    fc2 = mx.symbol.FullyConnected(data=tanh, num_hidden=num_classes)
    softmax = mx.symbol.SoftmaxOutput(data=fc2, name="softmax")
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

# create model
devs = mx.gpu(0)
network=get_symbol()
model = mx.model.FeedForward(
        ctx                = devs,
        symbol             = network,
        num_epoch          = 50,
        learning_rate      = 0.01,
        momentum           = 0.9,
        wd                 = 0.0001,
        initializer        = mx.init.Xavier(factor_type="in", magnitude=2.34)
        )

eval_metrics = ['accuracy']
model.fit(
	X=train_iter,
	eval_metric        = eval_metrics,
	eval_data	 = val_iter
	)

#predict
test = pd.read_csv("./test.csv").values
test_data = test.astype('float32')
test_data = np.array(test_data).reshape((-1, 1, 28, 28))
test_data[:]/= 256.0
test_iter = mx.io.NDArrayIter(test_data, batch_size=batch_size)

pred = model.predict(X = test_iter)
pred = np.argsort(pred)
np.savetxt('submission_lb_bn_v3.csv', np.c_[range(1,len(test)+1),pred[:,9]], delimiter=',', header = 'ImageId,Label', comments = '', fmt='%d')

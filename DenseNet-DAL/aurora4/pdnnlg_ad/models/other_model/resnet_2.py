# Copyright 2014    Yajie Miao    Carnegie Mellon University

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
# WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
# MERCHANTABLITY OR NON-INFRINGEMENT.
# See the Apache 2 License for the specific language governing permissions and
# limitations under the License.

import cPickle
import gzip
import os
import sys
import time
import collections

import numpy
import json

import theano
import theano.tensor as T

from io_func import smart_open
import lasagne
from lasagne.layers import Conv2DLayer as ConvLayer
#from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.layers import ElemwiseSumLayer
from lasagne.layers import InputLayer
from lasagne.layers import DenseLayer
from lasagne.layers import GlobalPoolLayer
from lasagne.layers import PadLayer
from lasagne.layers import ExpressionLayer
from lasagne.layers import NonlinearityLayer
from lasagne.layers import BatchNormLayer
from lasagne.nonlinearities import softmax, rectify
from lasagne.layers import batch_norm

class ConvLayer_Config(object):
    """configuration for a convolutional layer """

    def __init__(self, input_shape=(3,1,28,28), filter_shape=(2, 1, 5, 5),
                 poolsize=(1, 1), activation=T.tanh, output_shape=(3,1,28,28),
                 flatten = False):
        self.input_shape = input_shape
        self.filter_shape = filter_shape
        self.poolsize = pool_size
        self.output_shape = output_shape
        self.flatten = flatten

class layer_info(object):
    def __init__(self, layer_type='fc', filter_shape=(256, 1024, 3,3), num_params=2):
        self.type = layer_type
        self.filter_shape = filter_shape
        self.num_params = num_params

class ResNet(object):

    def __init__(self, numpy_rng, theano_rng=None, cfg = None, testing = False, input = None):

        self.layers = []
        self.params = []
        self.network = None

        self.cfg = cfg
        self.conv_layer_configs = cfg.conv_layer_configs
        self.conv_activation = cfg.conv_activation
        self.use_fast = cfg.use_fast

        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))
        # allocate symbolic variables for the data
        if input == None:
            self.x = T.matrix('x')
            self.input = T.tensor4('inputs')
        else:
            self.x = input
        self.y = T.ivector('y')

        self.conv_layer_num = len(self.conv_layer_configs)

        config = self.conv_layer_configs[0]
        d1 = 1
        d2 = 41
        d3 = 40
        self.input = self.x.reshape((-1,d1,d2,d3))
        print "[Debug] input_shape: ", d1, d2, d3

    
        self.network = lasagne.layers.InputLayer(shape=(None,d1,d2,d3), input_var=self.input)

        self.network = ConvLayer(self.network, num_filters=256, filter_size=(41,11), stride=(2,2), 
                        nonlinearity=None, pad='same', W=lasagne.init.HeNormal(gain='relu'), flip_filters=False)
        self.layers.append(layer_info('conv', (32, 1, 41, 11), 2)) # W, b
        
        n = 2
        # first stack of residual blocks, output is 256 x 256 x 20
        for _ in range(n):
            self.network = self.residual_block(self.network)
        
        # second stack of residual blocks, output is 512 x 512 x 10
        self.network = self.residual_block(self.network, increase_dim=True, projection=True)
        for _ in range(1,n):
            self.network = self.residual_block(self.network)
        
        # third stack of residual blocks, output is 1024 x 1024 x 5
        self.network = self.residual_block(self.network, increase_dim=True, projection=True)
        for _ in range(1,n):
            self.network = self.residual_block(self.network)
        
        # fourth stack of residual blocks, output is 1024 x 1024 x 8
        #self.network = self.residual_block(self.network, increase_dim=True, projection=True)
        #for _ in range(1,n):
        #    self.network = self.residual_block(self.network)

        # Average pool
        self.network = lasagne.layers.GlobalPoolLayer(self.network)        

        net_shape = lasagne.layers.get_output_shape(self.network)
        print "[Debug] net_shape: ", net_shape
        input_size = net_shape[1]
        self.conv_output_dim = net_shape[1] 
        cfg.n_ins = net_shape[1] 

        self.network = lasagne.layers.DenseLayer(self.network, num_units=self.cfg.n_outs, nonlinearity=None)
        self.layers.append(layer_info('fc', (input_size, self.cfg.n_outs, ), 2))
        self.network = lasagne.layers.NonlinearityLayer(self.network, lasagne.nonlinearities.softmax)

        # get params of each layer
        self.params = lasagne.layers.get_all_params(self.network, trainable=True)

        # define the cost and error
        pp = lasagne.layers.get_output(self.network)
        self.finetune_cost = -T.mean(T.log(pp)[T.arange(self.y.shape[0]), self.y])
        self.errors = T.mean(T.neq(T.argmax(lasagne.layers.get_output(self.network), axis=1), self.y))

    def residual_block(self, l, increase_dim=False, projection=False):
        # create a residual learning building block with two stacked 3x3 convlayers as in paper
        input_num_filters = l.output_shape[1]
        if increase_dim:
            first_stride = (2,2)
            out_num_filters = input_num_filters*2
        else:
            first_stride = (1,1)
            out_num_filters = input_num_filters

        l = BatchNormLayer(l)
        self.layers.append(layer_info('bn', num_params=4))

        l = NonlinearityLayer(l, nonlinearity=rectify)

        stack_1 = ConvLayer(l, num_filters=out_num_filters, filter_size=(3,3), stride=first_stride, nonlinearity=None, pad='same', W=lasagne.init.HeNormal(gain='relu'), flip_filters=False)
        self.layers.append(layer_info('conv', (out_num_filters, input_num_filters, 3, 3), 2))

        stack_1 = BatchNormLayer(stack_1)
        self.layers.append(layer_info('bn', num_params=4))

        stack_1 = NonlinearityLayer(stack_1, nonlinearity=rectify)

        stack_2 = ConvLayer(stack_1, num_filters=out_num_filters, filter_size=(3,3), stride=(1,1), nonlinearity=None, pad='same', W=lasagne.init.HeNormal(gain='relu'), flip_filters=False)
        self.layers.append(layer_info('conv', (out_num_filters, out_num_filters, 3, 3), 2))

        # add shortcut connections
        if increase_dim:
            if projection:
                # projection shortcut, as option B in paper
                projection = ConvLayer(l, num_filters=out_num_filters, filter_size=(1,1), stride=(2,2), nonlinearity=None, pad='same', b=None, flip_filters=False)
                self.layers.append(layer_info('conv-1', (out_num_filters, input_num_filters, 1, 1), 1))
                block = NonlinearityLayer(ElemwiseSumLayer([stack_2, projection]), nonlinearity=rectify) 
            else:
                # identity shortcut, as option A in paper
                identity = ExpressionLayer(l, lambda X: X[:, :, ::2, ::2], lambda s: (s[0], s[1], s[2]//2, s[3]//2))
                padding = PadLayer(identity, [out_num_filters//4,0,0], batch_ndim=1)
                block = NonlinearityLayer(ElemwiseSumLayer([stack_2, padding]), nonlinearity=rectify)
        else:
            block = NonlinearityLayer(ElemwiseSumLayer([stack_2, l]), nonlinearity=rectify)
        
        return block

    def build_finetune_functions(self, train_shared_xy, valid_shared_xy, batch_size):

        (train_set_x, train_set_y) = train_shared_xy
        (valid_set_x, valid_set_y) = valid_shared_xy

        print "[Debug] build_finetune_functions..."
        index = T.lscalar('index')  # index to a [mini]batch
        learning_rate = T.fscalar('learning_rate')

        prediction = lasagne.layers.get_output(self.network)
        y_pred = T.argmax(prediction, axis=1)
        acc = T.mean(T.eq(y_pred, self.y), dtype=theano.config.floatX)

        updates = lasagne.updates.sgd(self.finetune_cost, self.params, learning_rate)

        train_fn = theano.function(inputs=[index, learning_rate],
              outputs=[acc, self.errors],
              updates=updates,
              givens={
                self.x: train_set_x[index * batch_size:
                                    (index + 1) * batch_size],
                self.y: train_set_y[index * batch_size:
                                    (index + 1) * batch_size]})
 
        valid_fn = theano.function(inputs=[index],
              outputs=[acc, self.errors],
              givens={
                self.x: valid_set_x[index * batch_size:
                                    (index + 1) * batch_size],
                self.y: valid_set_y[index * batch_size:
                                    (index + 1) * batch_size]})

        return train_fn, valid_fn

    def build_extract_feat_function(self):

        feat = T.matrix('feat')

        layers = lasagne.layers.get_all_layers(self.network)
        print "[Debug] build_extract_feat_function"
        print "[Debug] layers: ", layers
        print "[Debug] len(layers) = ", len(layers)

        intermediate = lasagne.layers.get_output(layers[-3])
        output = intermediate.reshape((-1, self.conv_output_dim))
        out_da = theano.function([feat], output, updates = None, givens={self.x:feat}, on_unused_input='warn')
        return out_da

    def write_model_to_kaldi(self, file_path, with_softmax = True):
        # determine whether it's BNF based on layer sizes
        print "[Debug] write_model_to_kaldi"

        fout = smart_open(file_path, 'wb')
        params = lasagne.layers.get_all_params(self.network, trainable=True)
        activation_text = '<softmax>'
        W_mat = params[-2].get_value()
        b_vec = params[-1].get_value()
        input_size, output_size = W_mat.shape
        W_layer = []; b_layer = ''

        for rowX in xrange(output_size):
            W_layer.append('')
 
        for x in xrange(input_size):
            for t in xrange(output_size):
                W_layer[t] = W_layer[t] + str(W_mat[x][t]) + ' '

        for x in xrange(output_size):
            b_layer = b_layer + str(b_vec[x]) + ' '

        fout.write('<affinetransform> ' + str(output_size) + ' ' + str(input_size) + '\n')
        fout.write('[' + '\n')
        for x in xrange(output_size):
            fout.write(W_layer[x].strip() + '\n')
        fout.write(']' + '\n')
        fout.write('[ ' + b_layer.strip() + ' ]' + '\n')
        if activation_text == '<maxout>':
            fout.write(activation_text + ' ' + str(output_size/self.pool_size) + ' ' + str(output_size) + '\n')
        else:
            fout.write(activation_text + ' ' + str(output_size) + ' ' + str(output_size) + '\n')
        fout.close()

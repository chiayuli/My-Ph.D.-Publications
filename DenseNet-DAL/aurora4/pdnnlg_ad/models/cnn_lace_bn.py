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
from lasagne.nonlinearities import tanh, rectify, softmax
from lasagne.layers import InputLayer, DenseLayer, BatchNormLayer, NonlinearityLayer

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
        self.W = None
        self.b = None

    def set_filter_shape(self, shape):
        self.filter_shape = shape

class DotLayer(lasagne.layers.Layer):
    def __init__(self, incoming, num_units, W=lasagne.init.Normal(1), **kwargs):
        super(DotLayer, self).__init__(incoming, **kwargs)
        num_inputs = self.input_shape[1]
        self.num_units = num_units
        self.W = self.add_param(W, (num_inputs, num_units), name='W')
        #self.W = self.add_param(W, (5, 19), name='W')

    def get_output_for(self, input, **kwargs):
        p = numpy.multiply(input, self.W)
        print "[Debug] get_output_for: ", p.shape
        return numpy.multiply(input, self.W)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.num_units)

class CNN_LACE(object):

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
        d1 = config['input_shape'][1]
        d2 = config['input_shape'][2]
        d3 = config['input_shape'][3]
        self.input = self.x.reshape((-1,d1,d2,d3))
        #self.input = self.x.reshape((-1,d2,d3))

        num_chs = [128, 256, 512, 1024]
        self.network = lasagne.layers.InputLayer(shape=config['input_shape'], input_var=self.input)
        self.layers.append(layer_info('in_l', -1, 0))
        print "[In_l] params: ", lasagne.layers.get_all_params(self.network, trainable=True)

        num_JB = 4
        for i in range(num_JB):
            if i == 0:
                input_chs = 1
            else:
                input_chs = num_chs[i-1]
            
            self.network = self.JumpBlock(self.network, input_chs, num_chs[i])
            net_shape = lasagne.layers.get_output_shape(self.network)
            print "[Debug] JumpBlock ", i
            print "[Debug] net_shape: ", net_shape

        # Convolution (4x3, 1024 Channel)
        self.network = lasagne.layers.Conv2DLayer(self.network, num_filters=1024, filter_size=(4,3), 
                        stride=(2,2), nonlinearity=None)
        self.layers.append(layer_info('conv', (1024, 1024, 4, 3), 2)) # W and b
        P_array = lasagne.layers.get_all_params(self.network, trainable=True)
        print "[Debug-conv(final)] params: ", P_array

        net_shape = lasagne.layers.get_output_shape(self.network)
        print "[Debug] net_shape: ", net_shape

        self.conv_output_dim = net_shape[1] * net_shape[2] * net_shape[3] 
        cfg.n_ins = net_shape[1] * net_shape[2] * net_shape[3]


        print "[Debug] fc hidden layers sizes: ", len(self.cfg.hidden_layers_sizes)
        input_size = cfg.n_ins
        for i in range(len(self.cfg.hidden_layers_sizes)):
            if i == 0:
                input_size = cfg.n_ins
            else:
                input_size = self.cfg.hidden_layers_sizes[i-1]
            output_size = self.cfg.hidden_layers_sizes[i]

            print "[Debug] %d fc hidden layers sizes: %d" % (i, self.cfg.hidden_layers_sizes[i])
            self.network = lasagne.layers.DenseLayer(self.network, num_units=self.cfg.hidden_layers_sizes[i],
                                   nonlinearity=lasagne.nonlinearities.sigmoid)    
            self.layers.append(layer_info('fc', (input_size, self.cfg.hidden_layers_sizes[i]), 2))

        self.network = lasagne.layers.DenseLayer(self.network,
                                 num_units=self.cfg.n_outs, nonlinearity=lasagne.nonlinearities.softmax)
        self.layers.append(layer_info('fc', (input_size, self.cfg.n_outs), 2))

        # get params of each layer
        self.params = lasagne.layers.get_all_params(self.network, trainable=True)
        print "[softmax] params:", self.params

        # define the cost and error
        pp = lasagne.layers.get_output(self.network)
        self.finetune_cost = -T.mean(T.log(pp)[T.arange(self.y.shape[0]), self.y])
        self.errors = T.mean(T.neq(T.argmax(lasagne.layers.get_output(self.network), axis=1), self.y))

    def JumpBlock(self, prev_net, prev_num_chs, num_chs):
        print "[Debug] JumpBlock"
        curr_net = None

        # Convolution (Increase Channel + Reduce Width/Height)
        curr_net = lasagne.layers.Conv2DLayer(prev_net, num_filters=num_chs, filter_size=(3,3), 
                        stride=(2,2), pad='same', nonlinearity=None)
        self.layers.append(layer_info('conv', (num_chs, prev_num_chs, 3, 3), 2)) # W and b
        print "[conv (1)] params: ",  lasagne.layers.get_all_params(curr_net, trainable=True)

        # Multiple JumpNet
        J_net = None
        for i in range(2):
            if J_net is None:
                J_net = curr_net
            J_net = self.JumpNet(J_net, num_chs)

        return J_net      

    def JumpNet(self, net, num_chs):
        print "[Debug] JumpNet enter:"

        # Convolution (Keep Same Channel/Width/Height)  
        curr_net = lasagne.layers.Conv2DLayer(net, num_filters=num_chs, filter_size=(3,3), 
                        stride=1, pad='same', nonlinearity=lasagne.nonlinearities.rectify)
        self.layers.append(layer_info('conv', (num_chs, num_chs, 3, 3), 2))
        print "[conv (2)] params: ", lasagne.layers.get_all_params(curr_net, trainable=True)

        # Batch Normaalization and ReLU
        curr_net = BatchNormLayer(curr_net)
        self.layers.append(layer_info('bn', -1, 2)) # beta and gamma

        curr_net = NonlinearityLayer(curr_net, nonlinearity=rectify)
        self.layers.append(layer_info('relu', -1, 0))
        print "[bn_relu (3)] params: ", lasagne.layers.get_all_params(curr_net, trainable=True)

        # Convolution (Keep Same Channel/Width/Height)
        curr_net = lasagne.layers.Conv2DLayer(curr_net, num_filters=num_chs, filter_size=(3,3), 
                        stride=1, pad='same', nonlinearity=None)
        self.layers.append(layer_info('conv', (num_chs, num_chs, 3, 3), 2)) # W and b
        print "[conv (4)] params: ", lasagne.layers.get_all_params(curr_net, trainable=True)

        # Plus
        sum_net = lasagne.layers.ElemwiseSumLayer([net, curr_net])
        self.layers.append(layer_info('sum', -1, 0)) # none
        print "[sum (5)] params: ", lasagne.layers.get_all_params(sum_net, trainable=True)

        # Batch Normaalization and ReLU
        sum_net = BatchNormLayer(sum_net)
        self.layers.append(layer_info('bn', -1, 2)) # beta and gamma

        sum_net = NonlinearityLayer(sum_net, nonlinearity=rectify)
        self.layers.append(layer_info('relu', -1, 0))
        print "[bn_relu (6)] params: ", lasagne.layers.get_all_params(sum_net, trainable=True)

        return sum_net

    def build_finetune_functions(self, train_shared_xy, valid_shared_xy, batch_size):

        (train_set_x, train_set_y) = train_shared_xy
        (valid_set_x, valid_set_y) = valid_shared_xy

        print "[Debug] build_finetune_functions..."
        index = T.lscalar('index')  # index to a [mini]batch
        learning_rate = T.fscalar('learning_rate')

        prediction = lasagne.layers.get_output(self.network)
        y_pred = T.argmax(prediction, axis=1)
        acc = T.mean(T.eq(y_pred, self.y), dtype=theano.config.floatX)

        updates = lasagne.updates.adam(self.finetune_cost, self.params, learning_rate)

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

    def build_extract_feat_function(self, output_layer):
        print "[Debug] build_extract_feat_function"
        feat = T.matrix('feat')
        
        num_fc = len(self.cfg.hidden_layers_sizes) + 1
        layers = lasagne.layers.get_all_layers(self.network)
        print "[Debug] layers: ", layers
        print "[Debug] layers[-2] = ", layers[-2]

        intermediate = lasagne.layers.get_output(layers[-2])
        output = intermediate.reshape((-1, self.conv_output_dim))
        out_da = theano.function([feat], output, updates = None, givens={self.x:feat}, on_unused_input='warn')
        return out_da

    def write_model_to_kaldi(self, file_path, with_softmax = True):
        # determine whether it's BNF based on layer sizes
        print "[Debug] write_model_to_kaldi"

        fout = smart_open(file_path, 'wb')
        params_now = lasagne.layers.get_all_params(self.network, trainable=True)
        print "[Debug] params_now: ", params_now
        num_fc = len(self.cfg.hidden_layers_sizes) + 1
        start = len(params_now) - 2*num_fc
        end = len(params_now)
        print "[Debug] num_fc ", num_fc
        print "[Debug] start, end : ", start, end
        for i in range(start, end, 2):
            #if self.layers[i].type == 'fc':
            activation_text = '<' + self.cfg.activation_text + '>'
            if i == (end-2) and with_softmax:   # we assume that the last layer is a softmax layer
                activation_text = '<softmax>'
            W_mat = params_now[i].get_value()
            b_vec = params_now[i+1].get_value()
            input_size, output_size = W_mat.shape
            W_layer = [''] * output_size; b_layer = ''

            for t in xrange(output_size):
                b_layer = b_layer + str(b_vec[t]) + ' '
                W_layer[t] = ' '.join(map(str, W_mat[:, t])) + ' '

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

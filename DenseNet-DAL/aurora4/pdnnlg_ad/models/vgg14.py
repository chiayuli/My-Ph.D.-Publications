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
from lasagne.nonlinearities import rectify, softmax, sigmoid

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
    def __init__(self, layer_type='fc', filter_shape=(256, 2048, 3,3), num_params=2):
        self.type = layer_type
        self.filter_shape = filter_shape
        self.num_params = num_params
        self.W = None
        self.b = None

    def set_filter_shape(self, shape):
        self.filter_shape = shape

class VGG(object):

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
        print "[Debug] input_shape: ", d1, d2, d3

    
        self.network = lasagne.layers.InputLayer(shape=(None,d1,d2,d3), input_var=self.input)

        # 3 conv(64)
        for i in range(3):
            if i == 0:
                prev_num = d1
                filter=(1,3)
            else:
                prev_num = 64
                filter=(3,3)
            self.network = lasagne.layers.Conv2DLayer(self.network, num_filters=64, filter_size=filter, pad=1)
            self.layers.append(layer_info('conv', (64, prev_num, filter[0], filter[1]), 2)) # W and b
       
        # pool 1X3
        self.network = lasagne.layers.MaxPool2DLayer(self.network, pool_size=(1,2))
        #self.network = lasagne.layers.Pool2DLayer(self.network, (1,3), mode='average_inc_pad')

        # 4 conv(128)
        for i in range(4):
            if i == 0:
                prev_num = 64
            else:
                prev_num = 128
            self.network = lasagne.layers.Conv2DLayer(self.network, num_filters=128, filter_size=(3,3), pad=1)
            self.layers.append(layer_info('conv', (128, prev_num, 3, 3), 2)) # W and b
            print "[conv(128)] %d shape %s" %(i, lasagne.layers.get_output_shape(self.network))
       
        # pool 1X3
        self.network = lasagne.layers.MaxPool2DLayer(self.network, pool_size=(1,2), ignore_border=False)
        #self.network = lasagne.layers.Pool2DLayer(self.network, (1,3), mode='average_inc_pad')

        # 4 conv(256)
        for i in range(4):
            if i == 0:
                prev_num = 128
            else:
                prev_num = 256
            self.network = lasagne.layers.Conv2DLayer(self.network, num_filters=256, filter_size=(3,3), pad=1)
            self.layers.append(layer_info('conv', (256, prev_num, 3, 3), 2)) # W and b
        #    print "[conv(256)] %d shape %s" %(i, lasagne.layers.get_output_shape(self.network))
       
        # pool 1X3
        #self.network = lasagne.layers.MaxPool2DLayer(self.network, pool_size=(1,2), ignore_border=False)
        #self.network = lasagne.layers.Pool2DLayer(self.network, (1,3), mode='average_inc_pad')

        net_shape = lasagne.layers.get_output_shape(self.network)
        print "Final (before fc) shape. ", net_shape

        input_size = net_shape[1] * net_shape[2] * net_shape[3]
        self.conv_output_dim = net_shape[1] * net_shape[2] * net_shape[3]
        cfg.n_ins = net_shape[1] * net_shape[2] * net_shape[3]
        self.num_fc = 3

        self.network = lasagne.layers.DenseLayer(self.network, num_units=2048, nonlinearity=sigmoid)
        self.layers.append(layer_info('fc', (input_size, 2048), 2))
        #self.network = lasagne.layers.DropoutLayer(self.network, p=0.5)
        self.network = lasagne.layers.DenseLayer(self.network, num_units=2048, nonlinearity=sigmoid)
        self.layers.append(layer_info('fc', (2048, 2048), 2))
        #self.network = lasagne.layers.DropoutLayer(self.network, p=0.5)
        #self.network = lasagne.layers.DenseLayer(self.network, num_units=2048)
        #self.layers.append(layer_info('fc', (2048, 2048), 2))
        self.network = lasagne.layers.DenseLayer(self.network, num_units=self.cfg.n_outs, nonlinearity=softmax)
        self.layers.append(layer_info('fc', (2048, self.cfg.n_outs), 2))

        # get params of each layer
        self.params = lasagne.layers.get_all_params(self.network, trainable=True)

        # define the cost and error
        pp = lasagne.layers.get_output(self.network)
        self.finetune_cost = -T.mean(T.log(pp)[T.arange(self.y.shape[0]), self.y])
        self.errors = T.mean(T.neq(T.argmax(lasagne.layers.get_output(self.network), axis=1), self.y))

    def build_finetune_functions(self, train_shared_xy, valid_shared_xy, batch_size):

        (train_set_x, train_set_y) = train_shared_xy
        (valid_set_x, valid_set_y) = valid_shared_xy

        print "[Debug] build_finetune_functions..."
        index = T.lscalar('index')  # index to a [mini]batch
        learning_rate = T.fscalar('learning_rate')
        #momentum = T.fscalar('momentum')

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
        index = (-1 * self.num_fc) -1
        print "[Debug] layers[output_index]: ", layers[index]
        intermediate = lasagne.layers.get_output(layers[index])
        output = intermediate.reshape((-1, self.conv_output_dim))
        out_da = theano.function([feat], output, updates = None, givens={self.x:feat}, on_unused_input='warn')
        return out_da

    def write_model_to_kaldi(self, file_path, with_softmax = True):
        # determine whether it's BNF based on layer sizes
        print "[Debug] write_model_to_kaldi"

        fout = smart_open(file_path, 'wb')
        params_now = lasagne.layers.get_all_params(self.network, trainable=True)
        print "[Debug] params_now: ", params_now
        start = len(params_now) - (2 * self.num_fc)
        end = len(params_now)
        print "[Debug] num_fc ", self.num_fc
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


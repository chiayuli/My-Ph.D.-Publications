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

import json

import theano
import theano.tensor as T

from io_func import smart_open
import lasagne
from lasagne.layers import (InputLayer, Conv2DLayer, ConcatLayer, DenseLayer,
                             DropoutLayer, Pool2DLayer, GlobalPoolLayer,
                             NonlinearityLayer, batch_norm)

from lasagne.nonlinearities import rectify, softmax

DEBUG=False

class DENSENET(object):

    def __init__(self, cfg = None, testing = False, input = None):

        self.layers = []
        self.params = []
        self.network = None

        self.cfg = cfg
        #self.conv_layer_configs = cfg.conv_layer_configs

        if input == None:
            self.x = T.matrix('x')
        else:
            self.x = input
        
        self.y = T.ivector('y')

        #config = self.conv_layer_configs[0]
        #d1 = config['input_shape'][1]
        #d2 = config['input_shape'][2]
        #d3 = config['input_shape'][3]

        num_blocks = 3
        depth = 40
        growth_rate = 12
        dropout=0

    
        if DEBUG:
            print "[DEBUG] input_shape: ", config['input_shape']
            print "[DEBUG] DenseNet (num_blocks, depth, growth_rate, dropout): ", num_blocks, depth, growth_rate, dropout

        self.network = InputLayer(shape=(None,3,11,40), input_var=self.x.reshape((-1,3,11,40)))
        self.network = Conv2DLayer(self.network, num_filters=256, filter_size=3, 
                            W=lasagne.init.HeNormal(gain='relu'), b=None, nonlinearity=rectify)
        self.layers.append("conv-1") # W

        # note: The authors' implementation does *not* have a dropout after the
        #       initial convolution. This was missing in the paper, but important.
        # if dropout:
        #     network = DropoutLayer(network, dropout)
        # dense blocks with transitions in between

        n = (depth - 1) // num_blocks
        for b in range(num_blocks):
            self.network = self.dense_block(self.network, n - 1, growth_rate, dropout)
            if b < num_blocks - 1:
                self.network = self.transition(self.network, dropout)

            if DEBUG:
                print "[DEBUG] Dense Block %d network shape %s" %( b, self.network.output_shape )


        self.network = GlobalPoolLayer(self.network)

        self.conv_output_dim = self.network.output_shape[1]
        self.cfg.n_ins = self.network.output_shape[1]

        self.network = DenseLayer(self.network, self.cfg.n_outs, nonlinearity=softmax, W=lasagne.init.HeNormal(gain=1))
        self.layers.append("fc") # W and b

        self.params = lasagne.layers.get_all_params(self.network, trainable=True)

        # define the cost and error
        prediction = lasagne.layers.get_output(self.network)
        y_pred = T.argmax(prediction, axis=1)
        self.finetune_cost = lasagne.objectives.categorical_crossentropy(prediction, self.y).mean()
        self.acc = T.mean(T.eq(y_pred, self.y), dtype=theano.config.floatX)
        self.err = T.mean(T.neq(y_pred, self.y), dtype=theano.config.floatX)

    def dense_block(self, network, num_layers, growth_rate, dropout):
        # concatenated 3x3 convolutions
        for n in range(num_layers):
            conv = self.bn_relu_conv(network, channels=growth_rate, filter_size=3, dropout=dropout)
            network = ConcatLayer([network, conv], axis=1)

        return network


    def transition(self, network, dropout):
        # a transition 1x1 convolution followed by avg-pooling
        network = self.bn_relu_conv(network, channels=network.output_shape[1],
                               filter_size=1, dropout=dropout)
        network = Pool2DLayer(network, 2, mode='average_inc_pad')

        return network

    def bn_relu_conv(self, network, channels, filter_size, dropout):

        network = Conv2DLayer(network, channels, filter_size, pad='same', W=lasagne.init.HeNormal(gain='relu'),
                              b=None, nonlinearity=rectify)
        self.layers.append("conv-1") # W

        if dropout:
            network = DropoutLayer(network, dropout)
        return network


    def build_finetune_functions(self, train_shared_xy, valid_shared_xy, batch_size):

        (train_set_x, train_set_y) = train_shared_xy
        (valid_set_x, valid_set_y) = valid_shared_xy

        index = T.lscalar('index')  # index to a [mini]batch
        learning_rate = T.fscalar('learning_rate')

        l2_loss = 1e-4 * lasagne.regularization.regularize_network_params(
                self.network, lasagne.regularization.l2, {'trainable': True})
        updates = lasagne.updates.nesterov_momentum(
                self.finetune_cost + l2_loss, self.params, learning_rate, momentum=0.9)

        train_fn = theano.function(inputs=[index, learning_rate],
              outputs=[self.acc, self.err],
              updates=updates,
              givens={
                self.x: train_set_x[index * batch_size:
                                    (index + 1) * batch_size],
                self.y: train_set_y[index * batch_size:
                                    (index + 1) * batch_size]})
 
        valid_fn = theano.function(inputs=[index],
              outputs=[self.acc, self.err],
              givens={
                self.x: valid_set_x[index * batch_size:
                                    (index + 1) * batch_size],
                self.y: valid_set_y[index * batch_size:
                                    (index + 1) * batch_size]})

        return train_fn, valid_fn

    def build_extract_feat_function(self):

        feat = T.matrix('feat')

        layers = lasagne.layers.get_all_layers(self.network)
        
        intermediate = lasagne.layers.get_output(layers[-2])
        output = intermediate.reshape((-1, self.conv_output_dim))
        out_da = theano.function([feat], output, updates = None, givens={self.x:feat}, on_unused_input='warn')
        return out_da

    def write_model_to_kaldi(self, file_path, with_softmax = True):
        # determine whether it's BNF based on layer sizes
        print "[DEBUG] write_model_to_kaldi"

        fout = smart_open(file_path, 'wb')
        params = lasagne.layers.get_all_params(self.network, trainable=True)
        
        activation_text = '<softmax>'
        W_mat = params[-2].get_value()
        b_vec = params[-1].get_value()
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

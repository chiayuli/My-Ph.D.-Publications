# Copyright 2013    Yajie Miao    Carnegie Mellon University

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

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from io_func import smart_open
from io_func.model_io import _nnet2file, _file2nnet
import lasagne

class layer_info(object):
    def __init__(self, layer_type='fc', filter_shape=(3,1,28,28)):
        self.type = layer_type
        self.filter_shape = filter_shape

class DNN(object):

    def __init__(self, numpy_rng, theano_rng=None,
                 cfg = None,  # the network configuration
                 dnn_shared = None, shared_layers=[], input = None):

        self.layers = []
        self.params = []
        self.network = None

        self.cfg = cfg
        self.n_ins = cfg.n_ins; self.n_outs = cfg.n_outs
        self.hidden_layers_sizes = cfg.hidden_layers_sizes
        self.hidden_layers_number = len(self.hidden_layers_sizes)
        self.activation = cfg.activation

        self.do_maxout = cfg.do_maxout; self.pool_size = cfg.pool_size

        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))
        # allocate symbolic variables for the data
        if input == None:
            self.x = T.matrix('x')
        else:
            self.x = input
        self.y = T.ivector('y')

        # construct DNN
        self.network = lasagne.layers.InputLayer((None, self.n_ins), input_var=self.x)
        for i in range(0, self.hidden_layers_number):
             if i == 0:
                 input_size = self.n_ins
             else:
                 input_size = self.hidden_layers_sizes[i-1]
             output_size = self.hidden_layers_sizes[i]
             # initial weight and bias
             W_values = numpy.asarray(numpy_rng.uniform(
                    low=-numpy.sqrt(6. / (input_size + output_size)),
                    high=numpy.sqrt(6. / (output_size + output_size)),
                    size=(input_size, output_size)), dtype=theano.config.floatX)
             W_values *= 4
             b_values = numpy.zeros((output_size,), dtype=theano.config.floatX)

             self.network = lasagne.layers.DenseLayer(self.network, 
                                num_units=output_size, W=W_values, b=b_values, nonlinearity=lasagne.nonlinearities.sigmoid)
             self.layers.append(layer_info('fc'))
        # the last layer of DNN, activation is softmax
        self.network = lasagne.layers.DenseLayer(self.network, 
                           num_units=self.n_outs, nonlinearity=lasagne.nonlinearities.softmax)
        self.layers.append(layer_info('fc'))

        # get params of each layer
        params = lasagne.layers.get_all_params(self.network)
        for i in range(0, len(params), 2):
            tmp = [params[i], params[i+1]]
            self.params.extend(tmp)

        # define the cost and error
        pp = lasagne.layers.get_output(self.network)
        self.finetune_cost = -T.mean(T.log(pp)[T.arange(self.y.shape[0]), self.y])
        self.errors = T.mean(T.neq(T.argmax(lasagne.layers.get_output(self.network), axis=1), self.y))
        
    def build_finetune_functions(self, train_shared_xy, valid_shared_xy, batch_size):

        (train_set_x, train_set_y) = train_shared_xy
        (valid_set_x, valid_set_y) = valid_shared_xy

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


    def write_model_to_kaldi(self, file_path, with_softmax = True):
        # determine whether it's BNF based on layer sizes
        print "[Debug] write_model_to_kaldi"

        fout = smart_open(file_path, 'wb')
        params = lasagne.layers.get_all_params(self.network)
        for i in range(0, len(params), 2):
            activation_text = '<' + self.cfg.activation_text + '>'
            if i == (len(params)-2) and with_softmax:   # we assume that the last layer is a softmax layer
                activation_text = '<softmax>'
            W_mat = params[i].get_value()
            b_vec = params[i+1].get_value()
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


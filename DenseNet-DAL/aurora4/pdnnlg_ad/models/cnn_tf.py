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
    def __init__(self, layer_type='fc', filter_shape=(3,1,28,28)):
        self.type = layer_type
        self.filter_shape = filter_shape

class CNN(object):

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
        print "[Debug] input_shape: ", config['input_shape']
        print "[Debug] d1,d2,d3: ", d1, d2, d3

    
        self.network = lasagne.layers.InputLayer(shape=config['input_shape'], input_var=self.input)
        for i in xrange(self.conv_layer_num):
            config = self.conv_layer_configs[i]
            shape = (config['filter_shape'][2], config['filter_shape'][3])
            print "[Debug] %d - th layer" %i
            print "filter_shape: ", shape

            self.network = lasagne.layers.Conv2DLayer(self.network, num_filters=256, filter_size=shape,
                         nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.GlorotUniform())
            
            print "poolsize: ", config['poolsize']
            self.network = lasagne.layers.MaxPool2DLayer(self.network, pool_size=config['poolsize'])

            self.layers.append(layer_info('conv', config['filter_shape']))

        self.conv_output_dim = config['output_shape'][1] * config['output_shape'][2] * config['output_shape'][3]
        cfg.n_ins = config['output_shape'][1] * config['output_shape'][2] * config['output_shape'][3]
        print "[Debug] self.conv_output_dim: ", self.conv_output_dim
        print "[Debug] config['output_shape']: ", config['output_shape'][1], config['output_shape'][2], config['output_shape'][3]
        print "[Debug] cfg.n_ins: ", cfg.n_ins
      

        print "[Debug] fc hidden layers sizes: ", len(self.cfg.hidden_layers_sizes)
        for i in range(len(self.cfg.hidden_layers_sizes)):
            if i == 0:
                input_size = cfg.n_ins
            else:
                input_size = self.cfg.hidden_layers_sizes[i-1]
            output_size = self.cfg.hidden_layers_sizes[i]

            # initial weight and bias
            W_values = numpy.asarray(numpy_rng.uniform(
                   low=-numpy.sqrt(6. / (input_size + output_size)),
                   high=numpy.sqrt(6. / (output_size + output_size)),
                   size=(input_size, output_size)), dtype=theano.config.floatX)
            W_values *= 4
            b_values = numpy.zeros((output_size,), dtype=theano.config.floatX)

            print "[Debug] %d fc hidden layers sizes: %d" % (i, self.cfg.hidden_layers_sizes[i])
            self.network = lasagne.layers.DenseLayer(self.network, num_units=self.cfg.hidden_layers_sizes[i],
                                 W=W_values, b=b_values, nonlinearity=lasagne.nonlinearities.sigmoid)    
            self.layers.append(layer_info('fc'))

        self.network = lasagne.layers.DenseLayer(self.network,
                                 num_units=self.cfg.n_outs, nonlinearity=lasagne.nonlinearities.softmax)
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

    def build_extract_feat_function(self, output_layer):

        feat = T.matrix('feat')

        layers = lasagne.layers.get_all_layers(self.network)
        print "[Debug] build_extract_feat_function"
        print "[Debug] layers: ", layers
        print "[Debug] len(layers) = ", len(layers)
        if output_layer == 0:
            output_index = 2
        elif output_layer == 1:
            output_index = 4
        else:
            output_index = output_layer + 3

        print "[Debug] output_index: ", output_index
        intermediate = lasagne.layers.get_output(layers[output_index])
        output = intermediate.reshape((-1, self.conv_output_dim))
        out_da = theano.function([feat], output, updates = None, givens={self.x:feat}, on_unused_input='warn')
        return out_da

    def write_model_to_kaldi(self, file_path, with_softmax = True):
        # determine whether it's BNF based on layer sizes
        print "[Debug] write_model_to_kaldi"

        fout = smart_open(file_path, 'wb')
        params_now = lasagne.layers.get_all_params(self.network)
        print "[Debug] params_now: ", params_now
        num_layers = len(params_now)/2
        print "[Debug] len(self.layers): ", len(self.layers)
        for i in range(num_layers):
            if self.layers[i].type == 'fc':
                activation_text = '<' + self.cfg.activation_text + '>'
                if i == (num_layers-1) and with_softmax:   # we assume that the last layer is a softmax layer
                    activation_text = '<softmax>'
                W_mat = params_now[2*i].get_value()
                b_vec = params_now[2*i+1].get_value()
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

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
from io_func.model_io import _nnet2filex, _nnet2filey, _nnet2filez, _file2nnetx, _file2nnety, _file2nnetz
import lasagne
from lasagne.layers import (InputLayer, DenseLayer)
from lasagne.nonlinearities import LeakyRectify, sigmoid, softmax, rectify

class ReverseGradient(theano.Op):    
    """ theano operation to reverse the gradients    
    Introduced in http://arxiv.org/pdf/1409.7495.pdf    """    
    view_map = {0: [0]}    
    __props__ = ('hp_lambda', )    

    def __init__(self, hp_lambda):        
        super(ReverseGradient, self).__init__()        
        self.hp_lambda = hp_lambda    

    def make_node(self, x):        
        assert hasattr(self, '_props'), "Your version of theano is too old to support __props__."        
        x = theano.tensor.as_tensor_variable(x)        
        return theano.Apply(self, [x], [x.type()])    

    def perform(self, node, inputs, output_storage):        
        xin, = inputs        
        xout, = output_storage        
        xout[0] = xin    

    def grad(self, input, output_gradients):        
        return [-self.hp_lambda * output_gradients[0]]    

    def infer_shape(self, node, i0_shapes):        
        return i0_shapes

class ReverseGradientLayer(lasagne.layers.Layer):    
    def __init__(self, incoming, hp_lambda, **kwargs):        
        super(ReverseGradientLayer, self).__init__(incoming, **kwargs)        
        self.op = ReverseGradient(hp_lambda)    

    def get_output_for(self, input, **kwargs):        
        return self.op(input)

class DNN(object):

    def __init__(self, numpy_rng, theano_rng=None,
                 cfg = None,  # the network configuration
                 dnn_shared = None, shared_layers=[], input = None):

        self.layers = []
        self.params = []
        self.network = None
        self.epoch = 1

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
        self.z = T.ivector('z')
        self.intermediate = T.matrix('i')
        self.input = self.x.reshape((-1,3,11,40))

        self.x_network = self.build_x_network(input_var=self.input)        
        self.y_network = self.build_y_network(input_var=self.intermediate)        
        self.z_network = self.build_z_network(input_var=self.intermediate)

        # get params of each layer
        self.x_params = lasagne.layers.get_all_params(self.x_network)
        self.y_params = lasagne.layers.get_all_params(self.y_network)
        self.z_params = lasagne.layers.get_all_params(self.z_network)

        # define the cost and error
        pp = lasagne.layers.get_output(self.y_network, lasagne.layers.get_output(self.x_network))
        pp2 = lasagne.layers.get_output(self.z_network, lasagne.layers.get_output(self.x_network))
        y_pred = T.argmax(pp, axis=1)        
        y_pred2 = T.argmax(pp2, axis=1)        
        self.acc = T.mean(T.eq(y_pred, self.y), dtype=theano.config.floatX)
        self.acc2 = T.mean(T.eq(y_pred2, self.z), dtype=theano.config.floatX)
        self.y_finetune_cost = -T.mean(T.log(pp)[T.arange(self.y.shape[0]), self.y])
        self.z_finetune_cost = -T.mean(T.log(pp2)[T.arange(self.z.shape[0]), self.z])
        #alpha = min(self.epoch/10, 1) * 10
        self.x_finetune_cost = self.y_finetune_cost - 2*self.z_finetune_cost
        #self.epoch = self.epoch + 1
    
    def build_x_network(self, input_var=None):        
        layer = InputLayer(shape=(None, 3,11,40), input_var=input_var)        
        # fully-connected layer        
        #layer = DenseLayer(layer, 1024, nonlinearity=sigmoid)        
        #layer = DenseLayer(layer, 1024, nonlinearity=sigmoid)        
        layer = DenseLayer(layer, 1024, nonlinearity=sigmoid)        
        #layer = DenseLayer(layer, 1024, nonlinearity=sigmoid)        
        #layer = DenseLayer(layer, 1024, nonlinearity=sigmoid)        
        # project and reshape        
        #layer = DenseLayer(layer, 3*11*40, nonlinearity=None)        
        #layer = ReshapeLayer(layer, ([0], 3, 11, 40))        
        #print ("x_network output:", layer.output_shape)        
        return layer   
     
    def build_y_network(self, input_var=None):        
        layer = InputLayer(shape=(None, 1024), input_var=input_var)        
        # fully-connected layer        
        layer = DenseLayer(layer, 1024, nonlinearity=sigmoid)        
        layer = DenseLayer(layer, 1024, nonlinearity=sigmoid)        
        layer = DenseLayer(layer, 1024, nonlinearity=sigmoid)        
        layer = DenseLayer(layer, 1024, nonlinearity=sigmoid)        
        layer = DenseLayer(layer, 1024, nonlinearity=sigmoid)        
        layer = DenseLayer(layer, 1488, nonlinearity=softmax)        
        # project and reshape        
        #layer = DenseLayer(layer, 3*11*40, nonlinearity=None)        
        #layer = ReshapeLayer(layer, ([0], 3, 11, 40))        
        #print ("y_network output:", layer.output_shape)        
        return layer   
     
    def build_z_network(self, input_var=None):        
        layer = InputLayer(shape=(None, 1024), input_var=input_var)        
        # fully-connected layer        
        layer = ReverseGradientLayer(layer , 1)        
        #layer = DenseLayer(layer, 1024, nonlinearity=sigmoid)        
        layer = DenseLayer(layer, 512, nonlinearity=sigmoid)        
        #layer = DenseLayer(layer, 1024, nonlinearity=sigmoid)        
        layer = DenseLayer(layer, 4, nonlinearity=softmax)        
        # project and reshape        
        #layer = DenseLayer(layer, 3*11*40, nonlinearity=None)        
        #layer = ReshapeLayer(layer, ([0], 3, 11, 40))        
        #print ("z_network output:", layer.output_shape)        
        return layer   
     
    def build_finetune_functions(self, train_shared_xyz, valid_shared_xyz, batch_size):

        (train_set_x, train_set_y, train_set_z) = train_shared_xyz
        (valid_set_x, valid_set_y, valid_set_z) = valid_shared_xyz

        index = T.lscalar('index')  # index to a [mini]batch
        epoch = T.lscalar('epoch')  # index to a [mini]batch
        learning_rate = T.fscalar('learning_rate')
        #momentum = T.fscalar('momentum')

        updates_y = lasagne.updates.sgd(self.y_finetune_cost, self.y_params, learning_rate)
        updates_z = lasagne.updates.sgd(self.z_finetune_cost, self.z_params, learning_rate)        
        updates_x = lasagne.updates.sgd(self.x_finetune_cost, self.x_params, learning_rate)

        train_y_fn = theano.function(inputs=[index, learning_rate],
              outputs=[self.acc, self.y_finetune_cost],
              updates=updates_y,
              givens={
                self.x: train_set_x[index * batch_size:
                                    (index + 1) * batch_size],
                self.y: train_set_y[index * batch_size:
                                    (index + 1) * batch_size]})

        train_z_fn = theano.function(inputs=[index, learning_rate],
              outputs=[self.acc2, self.z_finetune_cost],
              updates=updates_z,
              givens={
                self.x: train_set_x[index * batch_size:
                                    (index + 1) * batch_size],
                self.z: train_set_z[index * batch_size:
                                    (index + 1) * batch_size]})

        train_x_fn = theano.function(inputs=[index, learning_rate],
              outputs=[self.acc, self.acc2],
              updates=updates_x,
              givens={
                self.x: train_set_x[index * batch_size:
                                    (index + 1) * batch_size],
                self.y: train_set_y[index * batch_size:
                                    (index + 1) * batch_size],
                self.z: train_set_z[index * batch_size:
                                    (index + 1) * batch_size]})
        
        valid_fn = theano.function(inputs=[index],
              outputs=[self.acc, self.acc2, 1-self.acc],
              givens={
                self.x: valid_set_x[index * batch_size:
                                    (index + 1) * batch_size],
                self.y: valid_set_y[index * batch_size:
                                    (index + 1) * batch_size],
                self.z: valid_set_z[index * batch_size:
                                    (index + 1) * batch_size]})

        
        return train_y_fn, train_z_fn, train_x_fn, valid_fn


    def write_model_to_kaldi(self, file_path, with_softmax = True):
        # determine whether it's BNF based on layer sizes
        print "[Debug] write_model_to_kaldi"

        fout = smart_open(file_path, 'wb')
        params = lasagne.layers.get_all_params(self.x_network)
        params += lasagne.layers.get_all_params(self.y_network)
        print "[Debug] params", params
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


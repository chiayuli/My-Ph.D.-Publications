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
from lasagne.layers import (InputLayer, Conv2DLayer, ConcatLayer, DenseLayer,
                             DropoutLayer, Pool2DLayer, GlobalPoolLayer,
                             NonlinearityLayer, batch_norm)

from lasagne.nonlinearities import rectify, softmax, sigmoid
#try:
#    from lasagne.layers.dnn import BatchNormDNNLayer as BatchNormLayer
#except ImportError:
from lasagne.layers import BatchNormLayer

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

class DENSENET(object):

    def __init__(self, cfg = None, testing = False, input = None):

        self.x_layers = []
        self.y_layers = []
        self.z_layers = []
        self.params = []
        self.network = None
        self.epoch = 1

        self.cfg = cfg
        self.conv_layer_configs = cfg.conv_layer_configs
        self.conv_activation = cfg.conv_activation
        self.use_fast = cfg.use_fast
        self.growth_rate = 12
        self.dropout=0        
        self.theta=0.4
        # allocate symbolic variables for the data
        if input == None:
            self.x = T.matrix('x')
            self.input = T.tensor4('inputs')
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
        alpha = min(self.epoch/10, 1) * 10        
        self.x_finetune_cost = self.y_finetune_cost - alpha*self.z_finetune_cost        
        self.epoch = self.epoch + 1

        #self.conv_layer_num = len(self.conv_layer_configs)

    def build_x_network(self, input_var=None):                
        layer = InputLayer(shape=(None, 3,11,40), input_var=input_var)
        num_blocks = 3
        depth = 46
        growth_rate = 12
    
        layer= lasagne.layers.Conv2DLayer(layer, num_filters=12, filter_size=3,
                            W=lasagne.init.HeNormal(gain='relu'),
                            b=None, nonlinearity=rectify)
        #print "pre_conv Params:", lasagne.layers.get_all_params(self.network, trainable=True)
        self.x_layers.append(layer_info('conv', (12, 3, 3, 3), 1)) # W
        #self.x_layers.append(layer_info('bn', num_params=2))
        # note: The authors' implementation does *not* have a dropout after the
        #       initial convolution. This was missing in the paper, but important.
        # if dropout:
        #     network = DropoutLayer(network, dropout)
        # dense blocks with transitions in between
        n = (depth - 1) // num_blocks
        for b in range(num_blocks):
            layer = self.dense_block(layer, n - 1, self.growth_rate, self.dropout, self.x_layers)
            #if b < num_blocks - 1:
            t_n = int(layer.output_shape[1] * self.theta)
            layer = self.transition(layer, t_n, self.dropout, self.x_layers)

            net_shape_tmp = lasagne.layers.get_output_shape(layer)
            print "[Debug] net_shape: ", b, net_shape_tmp
        return layer

    def build_y_network(self, input_var=None):
        input = input_var.reshape((-1,105,1,4))
        layer = InputLayer(shape=(None, 105, 1, 4), input_var=input)
        layer = self.dense_block(layer, 14, self.growth_rate, self.dropout, self.y_layers)
        layer = GlobalPoolLayer(layer)
        self.y_layers.append(layer_info('pool'))

        net_shape = lasagne.layers.get_output_shape(layer)
        print "[Debug] before fc, net_shape: ", net_shape

        self.conv_output_dim = net_shape[1]
        self.cfg.n_ins = net_shape[1]


        layer = DenseLayer(layer, self.cfg.n_outs, nonlinearity=softmax,
                             W=lasagne.init.HeNormal(gain=1))

        self.y_layers.append(layer_info('fc', (self.cfg.n_ins, self.cfg.n_outs), 2))
        return layer

    def build_z_network(self, input_var=None):                
        layer = InputLayer(shape=(None, 420), input_var=input_var)                
        # fully-connected layer                
        layer = ReverseGradientLayer(layer , 1)                
        #layer = DenseLayer(layer, 1024, nonlinearity=sigmoid)                
        layer = DenseLayer(layer, 1024, nonlinearity=sigmoid)                
        self.z_layers.append(layer_info('fc', (420, 1024), 2))
        #layer = DenseLayer(layer, 1024, nonlinearity=sigmoid)                
        layer = DenseLayer(layer, 4, nonlinearity=softmax)                
        self.z_layers.append(layer_info('fc', (1024, 4), 2))
        # project and reshape                
        #layer = DenseLayer(layer, 3*11*40, nonlinearity=None)                
        #layer = ReshapeLayer(layer, ([0], 3, 11, 40))                
        #print ("z_network output:", layer.output_shape)                
        return layer   

    def dense_block(self, network, num_layers, growth_rate, dropout, layers):
        # concatenated 3x3 convolutions
        for n in range(num_layers):
            conv = self.bn_relu_conv(network, channels=growth_rate,
                                filter_size=3, dropout=dropout, layers=layers)
            network = ConcatLayer([network, conv], axis=1)
            layers.append(layer_info('concat'))

        return network


    def transition(self, network, channels, dropout, layers):
        # a transition 1x1 convolution followed by avg-pooling
        print "network shape before transition: ", network.output_shape
        print "transition channels: ", channels
        network = self.bn_relu_conv(network, channels=channels,
                               filter_size=3, dropout=dropout, layers=layers)
        network = Pool2DLayer(network, 2, mode='average_inc_pad')
        layers.append(layer_info('pool')) 

        print "network shape after transition: ", network.output_shape
        return network

    def bn_relu_conv(self, network, channels, filter_size, dropout, layers):

        network = Conv2DLayer(network, channels, filter_size, pad='same',
                              W=lasagne.init.HeNormal(gain='relu'),
                              b=None, nonlinearity=rectify)
        layers.append(layer_info('conv', (channels, channels, 3, 3), 1))
        #layers.append(layer_info('bn', num_params=2))

        if dropout:
            network = DropoutLayer(network, dropout)
        return network


    def build_finetune_functions(self, train_shared_xyz, valid_shared_xyz, batch_size):

        (train_set_x, train_set_y, train_set_z) = train_shared_xyz
        (valid_set_x, valid_set_y, valid_set_z) = valid_shared_xyz

        print "[Debug] build_finetune_functions..."
        index = T.lscalar('index')  # index to a [mini]batch
        learning_rate = T.fscalar('learning_rate')


        #updates = lasagne.updates.nesterov_momentum(
        #        self.finetune_cost + l2_loss, self.params, learning_rate, momentum=0.9)
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

    def build_extract_feat_function(self, output_layer):

        feat = T.matrix('feat')
        
        layers = lasagne.layers.get_all_layers(self.y_network)
        print "[Debug] build_extract_feat_function"
        print "[Debug] layers: ", layers
        print "[Debug] layers[-2] = ", layers[-2]

        intermediate = lasagne.layers.get_output(self.y_network[-2], lasagne.layers.get_output(self.x_network))
        #intermediate = lasagne.layers.get_output(layers[-2])
        output = intermediate.reshape((-1, self.conv_output_dim))
        out_da = theano.function([feat], output, updates = None, givens={self.x:feat}, on_unused_input='warn')
        return out_da

    def write_model_to_kaldi(self, file_path, with_softmax = True):
        # determine whether it's BNF based on layer sizes
        print "[Debug] write_model_to_kaldi"

        fout = smart_open(file_path, 'wb')
        params = lasagne.layers.get_all_params(self.x_network, trainable=True)        
        params += lasagne.layers.get_all_params(self.y_network, trainable=True)
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

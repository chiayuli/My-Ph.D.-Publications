# Copyright 2013    Yajie Miao    Carnegie Mellon University
#           2015    Yun Wang      Carnegie Mellon University

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

# Various functions to write models from nets to files, and to read models from
# files to nets

import numpy as np
import os
import sys
import cPickle

from StringIO import StringIO
import json

import theano
import theano.tensor as T

from datetime import datetime

from io_func import smart_open

import lasagne

# print log to standard output
def log(string):
    sys.stderr.write('[' + str(datetime.now()) + '] ' + str(string) + '\n')

# convert an array to a string
def array_2_string(array):
    str_out = StringIO()
    np.savetxt(str_out, array)
    return str_out.getvalue()

# convert a string to an array
#def string_2_array(string):
#    str_in = StringIO(string)
#    return np.loadtxt(str_in)

def string_2_array(string):
    str_in = StringIO(string)
    array_tmp = np.loadtxt(str_in)
    if len(array_tmp.shape) == 0:
        return np.array([array_tmp])
    return array_tmp

def _nnet2filex(model, set_layer_num = -1, filename='nnet.out', start_layer = 0, input_factor = 0.0, factor=[]):
    #print "[Debug] _nnet2file"
    nnet_dict = {}
    pp = []
    params = lasagne.layers.get_all_params(model.x_network, trainable=True)
    #layers = lasagne.layers.get_all_layers(model.x_network)
    layer = model.x_layers
    for i in range(len(params)):
       pp.append(params[i].get_value())

    index = 0
    for i in range(len(layer)):
       dropout_factor = 0.0
       if layer[i].type == 'fc':
           # Have 2-D Weight and Bias
           #print "[Debug-fc] w.shape: ", pp[index].shape
           dict_a = 'W' + str(i)
           nnet_dict[dict_a] = array_2_string((1.0 - dropout_factor) * pp[index])
           index += 1
           dict_a = 'b' + str(i)
           nnet_dict[dict_a] = array_2_string(pp[index])
           index += 1
           
       elif layer[i].type == 'conv':
           # conv.W
           dict_a = 'W' + str(i)
           for next_X in xrange(pp[index].shape[0]):
               for this_X in xrange(pp[index].shape[1]):
                   new_dict_a = dict_a + ' ' + str(next_X) + ' ' + str(this_X)
                   nnet_dict[new_dict_a] = array_2_string(pp[index][next_X, this_X])
           index += 1

       elif layer[i].type == 'bn':
           dict_a = 'beta' + str(i)
           nnet_dict[dict_a] = array_2_string(pp[index])
           index += 1

           dict_a = 'gamma' + str(i)
           nnet_dict[dict_a] = array_2_string(pp[index])
           index += 1

    with smart_open(filename, 'wb') as fp:
        json.dump(nnet_dict, fp, indent=2, sort_keys = True)
        fp.flush()

def _nnet2filey(model, set_layer_num = -1, filename='nnet.out', start_layer = 0, input_factor = 0.0, factor=[]):
    #print "[Debug] _nnet2file"
    nnet_dict = {}
    pp = []
    params = lasagne.layers.get_all_params(model.y_network, trainable=True)
    #layers = lasagne.layers.get_all_layers(model.y_network)
    layer = model.y_layers
    for i in range(len(params)):
       pp.append(params[i].get_value())

    index = 0
    for i in range(len(layer)):
       dropout_factor = 0.0
       if layer[i].type == 'fc':
           # Have 2-D Weight and Bias
           #print "[Debug-fc] w.shape: ", pp[index].shape
           dict_a = 'W' + str(i)
           nnet_dict[dict_a] = array_2_string((1.0 - dropout_factor) * pp[index])
           index += 1
           dict_a = 'b' + str(i)
           nnet_dict[dict_a] = array_2_string(pp[index])
           index += 1
           
       elif layer[i].type == 'conv':
           # conv.W
           dict_a = 'W' + str(i)
           for next_X in xrange(pp[index].shape[0]):
               for this_X in xrange(pp[index].shape[1]):
                   new_dict_a = dict_a + ' ' + str(next_X) + ' ' + str(this_X)
                   nnet_dict[new_dict_a] = array_2_string(pp[index][next_X, this_X])
           index += 1

       elif layer[i].type == 'bn':
           dict_a = 'beta' + str(i)
           nnet_dict[dict_a] = array_2_string(pp[index])
           index += 1

           dict_a = 'gamma' + str(i)
           nnet_dict[dict_a] = array_2_string(pp[index])
           index += 1

    with smart_open(filename, 'wb') as fp:
        json.dump(nnet_dict, fp, indent=2, sort_keys = True)
        fp.flush()

def _nnet2filez(model, set_layer_num = -1, filename='nnet.out', start_layer = 0, input_factor = 0.0, factor=[]):
    #print "[Debug] _nnet2file"
    nnet_dict = {}
    pp = []
    params = lasagne.layers.get_all_params(model.z_network, trainable=True)
    #layers = lasagne.layers.get_all_layers(model.z_network)
    layer = model.z_layers
    for i in range(len(params)):
       pp.append(params[i].get_value())

    index = 0
    for i in range(len(layer)):
       dropout_factor = 0.0
       if layer[i].type == 'fc':
           # Have 2-D Weight and Bias
           #print "[Debug-fc] w.shape: ", pp[index].shape
           dict_a = 'W' + str(i)
           nnet_dict[dict_a] = array_2_string((1.0 - dropout_factor) * pp[index])
           index += 1
           dict_a = 'b' + str(i)
           nnet_dict[dict_a] = array_2_string(pp[index])
           index += 1
           
       elif layer[i].type == 'conv':
           # conv.W
           dict_a = 'W' + str(i)
           for next_X in xrange(pp[index].shape[0]):
               for this_X in xrange(pp[index].shape[1]):
                   new_dict_a = dict_a + ' ' + str(next_X) + ' ' + str(this_X)
                   nnet_dict[new_dict_a] = array_2_string(pp[index][next_X, this_X])
           index += 1

       elif layer[i].type == 'bn':
           dict_a = 'beta' + str(i)
           nnet_dict[dict_a] = array_2_string(pp[index])
           index += 1

           dict_a = 'gamma' + str(i)
           nnet_dict[dict_a] = array_2_string(pp[index])
           index += 1

    with smart_open(filename, 'wb') as fp:
        json.dump(nnet_dict, fp, indent=2, sort_keys = True)
        fp.flush()

def _nnet2filexy(model, set_layer_num = -1, filename='nnet.out', start_layer = 0, input_factor = 0.0, factor=[]):
    #print "[Debug] _nnet2file"
    nnet_dict = {}
    pp = []
    params = lasagne.layers.get_all_params(model.x_network, trainable=True)
    params += lasagne.layers.get_all_params(model.y_network, trainable=True)
    #layers = lasagne.layers.get_all_layers(model.x_network)
    #layers += lasagne.layers.get_all_layers(model.y_network)
    layer = model.x_layers
    layer += model.y_layers
    
    for i in range(len(params)):
       pp.append(params[i].get_value())

    index = 0
    for i in range(len(layer)):
       dropout_factor = 0.0
       if layer[i].type == 'fc':
           # Have 2-D Weight and Bias
           #print "[Debug-fc] w.shape: ", pp[index].shape
           dict_a = 'W' + str(i)
           nnet_dict[dict_a] = array_2_string((1.0 - dropout_factor) * pp[index])
           index += 1
           dict_a = 'b' + str(i)
           nnet_dict[dict_a] = array_2_string(pp[index])
           index += 1
           
       elif layer[i].type == 'conv':
           # conv.W
           dict_a = 'W' + str(i)
           for next_X in xrange(pp[index].shape[0]):
               for this_X in xrange(pp[index].shape[1]):
                   new_dict_a = dict_a + ' ' + str(next_X) + ' ' + str(this_X)
                   nnet_dict[new_dict_a] = array_2_string(pp[index][next_X, this_X])
           index += 1

       elif layer[i].type == 'bn':
           dict_a = 'beta' + str(i)
           nnet_dict[dict_a] = array_2_string(pp[index])
           index += 1

           dict_a = 'gamma' + str(i)
           nnet_dict[dict_a] = array_2_string(pp[index])
           index += 1

    with smart_open(filename, 'wb') as fp:
        json.dump(nnet_dict, fp, indent=2, sort_keys = True)
        fp.flush()

# save the config classes; since we are using pickle to serialize the whole class, it's better to set the
# data reading and learning rate interfaces to None.
def _cfg2file(cfg, filename='cfg.out'):
    cfg.lrate = None
    cfg.train_sets = None; cfg.train_xyz = None; cfg.train_x = None; cfg.train_y = None; cfg.train_z = None
    cfg.valid_sets = None; cfg.valid_xyz = None; cfg.valid_x = None; cfg.valid_y = None; cfg.valid_z = None
    cfg.activation = None  # saving the rectifier function causes errors; thus we don't save the activation function
                           # the activation function is initialized from the activation text ("sigmoid") when the network
                           # configuration is loaded
    with smart_open(filename, "wb") as output:
        cPickle.dump(cfg, output, cPickle.HIGHEST_PROTOCOL)

def _file2nnetx(model, set_layer_num = -1, filename='nnet.in',  factor=1.0):
    print "[Debug] _file2nnetx"
    nnet_dict = {}
    old_params = lasagne.layers.get_all_params(model.x_network, trainable=True)
    print "old_params: ", old_params
    #layers = lasagne.layers.get_all_layers(model.x_network)
    layer = model.x_layers
    new_params = []
    index = 0
    with smart_open(filename, 'rb') as fp:
        nnet_dict = json.load(fp)

    for i in xrange(len(layer)):

        if layer[i].type == 'fc':
            dict_a = 'W' + str(i)
            #print "[fc] ", dict_a
            mat_shape = old_params[index].get_value().shape
            #print "[Debug-fc] mat_shape ", mat_shape, layer.filter_shape
            W_mat = factor * np.asarray(string_2_array(nnet_dict[dict_a]), dtype=theano.config.floatX).reshape(mat_shape)
            new_params.append(W_mat)
            dict_a = 'b' + str(i)
            b_vec = np.asarray(string_2_array(nnet_dict[dict_a]), dtype=theano.config.floatX)
            new_params.append(b_vec)
            index += 2

        elif layer[i].type == 'conv':
            dict_a = 'W' + str(i)
            W_array = old_params[index].get_value()
            
            for next_X in xrange(W_array.shape[0]):
                for this_X in xrange(W_array.shape[1]):
                    new_dict_a = dict_a + ' ' + str(next_X) + ' ' + str(this_X)
                    mat_shape = W_array[next_X, this_X, :, :].shape
                    W_array[next_X, this_X, :, :] = factor * np.asarray(string_2_array(nnet_dict[new_dict_a]), dtype=theano.config.floatX).reshape(mat_shape)
            new_params.append(W_array)
            index += 1

        elif layer[i].type == 'bn':
            dict_a = 'beta' + str(i)
            b_vec = np.asarray(string_2_array(nnet_dict[dict_a]), dtype=theano.config.floatX)
            new_params.append(b_vec)

            dict_a = 'gamma' + str(i)
            b_vec = np.asarray(string_2_array(nnet_dict[dict_a]), dtype=theano.config.floatX)
            new_params.append(b_vec)

            index += 2

    lasagne.layers.set_all_param_values(model.x_network, new_params, trainable=True)

def _file2nnety(model, set_layer_num = -1, filename='nnet.in',  factor=1.0):
    print "[Debug] _file2nnety"
    nnet_dict = {}
    old_params = lasagne.layers.get_all_params(model.y_network, trainable=True)
    layer = model.y_layers
    print "---layer", layer
    print "---old_params", old_params
    new_params = []
    index = 0
    with smart_open(filename, 'rb') as fp:
        nnet_dict = json.load(fp)

    for i in xrange(len(layer)):

        if layer[i].type == 'fc':
            dict_a = 'W' + str(i)
            #print "[fc] ", dict_a
            mat_shape = old_params[index].get_value().shape
            #print "[Debug-fc] mat_shape ", mat_shape, layer.filter_shape
            W_mat = factor * np.asarray(string_2_array(nnet_dict[dict_a]), dtype=theano.config.floatX).reshape(mat_shape)
            new_params.append(W_mat)
            dict_a = 'b' + str(i)
            b_vec = np.asarray(string_2_array(nnet_dict[dict_a]), dtype=theano.config.floatX)
            new_params.append(b_vec)
            index += 2

        elif layer[i].type == 'conv':
            dict_a = 'W' + str(i)
            W_array = old_params[index].get_value()
            print "i W_array.shape: ", i, W_array.shape
            for next_X in xrange(W_array.shape[0]):
                for this_X in xrange(W_array.shape[1]):
                    new_dict_a = dict_a + ' ' + str(next_X) + ' ' + str(this_X)
                    mat_shape = W_array[next_X, this_X, :, :].shape
                    W_array[next_X, this_X, :, :] = factor * np.asarray(string_2_array(nnet_dict[new_dict_a]), dtype=theano.config.floatX).reshape(mat_shape)
            new_params.append(W_array)
            index += 1

        elif layer[i].type == 'bn':
            dict_a = 'beta' + str(i)
            b_vec = np.asarray(string_2_array(nnet_dict[dict_a]), dtype=theano.config.floatX)
            new_params.append(b_vec)

            dict_a = 'gamma' + str(i)
            b_vec = np.asarray(string_2_array(nnet_dict[dict_a]), dtype=theano.config.floatX)
            new_params.append(b_vec)

            index += 2

    lasagne.layers.set_all_param_values(model.y_network, new_params, trainable=True)

def _file2nnetz(model, set_layer_num = -1, filename='nnet.in',  factor=1.0):
    print "[Debug] _file2nnetz"
    nnet_dict = {}
    old_params = lasagne.layers.get_all_params(model.z_network, trainable=True)
    layer = model.z_layers
    new_params = []
    index = 0
    with smart_open(filename, 'rb') as fp:
        nnet_dict = json.load(fp)

    for i in xrange(len(layers)):

        if layer[i].type == 'fc':
            dict_a = 'W' + str(i)
            mat_shape = old_params[index].get_value().shape
            W_mat = factor * np.asarray(string_2_array(nnet_dict[dict_a]), dtype=theano.config.floatX).reshape(mat_shape)
            new_params.append(W_mat)
            dict_a = 'b' + str(i)
            b_vec = np.asarray(string_2_array(nnet_dict[dict_a]), dtype=theano.config.floatX)
            new_params.append(b_vec)
            index += 2

        elif layer[i].type == 'conv':
            dict_a = 'W' + str(i)
            W_array = old_params[index].get_value()
            
            for next_X in xrange(W_array.shape[0]):
                for this_X in xrange(W_array.shape[1]):
                    new_dict_a = dict_a + ' ' + str(next_X) + ' ' + str(this_X)
                    mat_shape = W_array[next_X, this_X, :, :].shape
                    W_array[next_X, this_X, :, :] = factor * np.asarray(string_2_array(nnet_dict[new_dict_a]), dtype=theano.config.floatX).reshape(mat_shape)
            new_params.append(W_array)
            index += 1

        elif layer[i].type == 'bn':
            dict_a = 'beta' + str(i)
            b_vec = np.asarray(string_2_array(nnet_dict[dict_a]), dtype=theano.config.floatX)
            new_params.append(b_vec)

            dict_a = 'gamma' + str(i)
            b_vec = np.asarray(string_2_array(nnet_dict[dict_a]), dtype=theano.config.floatX)
            new_params.append(b_vec)

            index += 2

    lasagne.layers.set_all_param_values(model.z_network, new_params, trainable=True)


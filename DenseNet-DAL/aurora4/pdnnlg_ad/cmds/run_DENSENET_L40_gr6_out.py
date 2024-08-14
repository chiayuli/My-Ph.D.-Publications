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

import numpy

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from io_func.model_io_densenet import _nnet2file, _cfg2file, _file2nnet, log
from utils.utils import parse_arguments
from utils.learn_rates import _lrate2file, _file2lrate

from utils.network_config import NetworkConfig
from learning.sgd import train_sgd, validate_by_minibatch

from models.densenet_L40_gr6 import DENSENET

if __name__ == '__main__':

    # check the arguments
    arg_elements = [sys.argv[i] for i in range(1, len(sys.argv))]
    arguments = parse_arguments(arg_elements)
    required_arguments = ['train_data', 'valid_data', 'nnet_spec', 'conv_nnet_spec', 'wdir']
    for arg in required_arguments:
        if arguments.has_key(arg) == False:
            print "Error: the argument %s has to be specified" % (arg); exit(1)

    # mandatory arguments
    train_data_spec = arguments['train_data']
    valid_data_spec = arguments['valid_data']
    conv_nnet_spec = arguments['conv_nnet_spec']
    nnet_spec = arguments['nnet_spec']
    wdir = arguments['wdir']

    # parse network configuration from arguments, and initialize data reading
    cfg = NetworkConfig(); cfg.model_type = 'CNN'
    cfg.parse_config_cnn(arguments, '10:' + nnet_spec, conv_nnet_spec)
    cfg.init_data_reading(train_data_spec, valid_data_spec)

    # check working dir to see whether it's resuming training
    resume_training = False
    if os.path.exists(wdir + '/nnet.tmp_DENSENET.L40_gr6.1') and os.path.exists(wdir + '/training_state.tmp_DENSENET.L40_gr6.1'):
        resume_training = True
        cfg.lrate = _file2lrate(wdir + '/training_state.tmp_DENSENET.L40_gr6.1')
        log('> ... found nnet.tmp_DENSENET.L40_gr6.1 and training_state.tmp_DENSENET.L40_gr6.1, now resume training from epoch ' + str(cfg.lrate.epoch))

    numpy_rng = numpy.random.RandomState(89677)
    theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))
    log('> ... initializing the model')
    # construct the cnn architecture
    densenet = DENSENET(numpy_rng=numpy_rng, theano_rng = theano_rng, cfg = cfg)
    # load the pre-training networks, if any, for parameter initialization
    if resume_training:
        _file2nnet(densenet, filename = wdir + '/nnet.tmp_DENSENET.L40_gr6.1')

    # get the training, validation and testing function for the model
    log('> ... getting the finetuning functions')
    train_fn, valid_fn = densenet.build_finetune_functions(
                (cfg.train_x, cfg.train_y), (cfg.valid_x, cfg.valid_y),
                batch_size=cfg.batch_size)

    log('> ... finetunning the model')
    #while (cfg.lrate.get_rate() != 0):
    if 1 == 0:
        lrate = 0.1
        if cfg.lrate.epoch > 4:
            lrate = 0.01
        elif cfg.lrate.epoch > 6:
            lrate = 0.001
        lrate = cfg.lrate.get_rate()
        # one epoch of sgd training
        train_acc = train_sgd(train_fn, cfg)
        log('> epoch %d, lrate %f, training accuracy %f' % (cfg.lrate.epoch, lrate, 100*numpy.mean(train_acc)) + '(%)')
        # validation 
        valid_acc, valid_error = validate_by_minibatch(valid_fn, cfg)
        log('> epoch %d, lrate %f, validate accuracy %f' % (cfg.lrate.epoch, lrate, 100*numpy.mean(valid_acc)) + '(%)')
        cfg.lrate.get_next_rate(current_error = 100 * numpy.mean(valid_error))

        # output nnet parameters and lrate, for training resume
        if cfg.lrate.epoch % cfg.model_save_step == 0:
            _nnet2file(densenet, filename=wdir + '/nnet.tmp_DENSENET.L40_gr6.1')
            _lrate2file(cfg.lrate, wdir + '/training_state.tmp_DENSENET.L40_gr6.1')

    # save the model and network configuration
    if cfg.param_output_file != '':
        _nnet2file(densenet, filename=cfg.param_output_file, input_factor = cfg.input_dropout_factor, factor = cfg.dropout_factor)
        log('> ... the final PDNN model parameter is ' + cfg.param_output_file)
    if cfg.cfg_output_file != '':
        _cfg2file(densenet.cfg, filename=cfg.cfg_output_file)
        log('> ... the final PDNN model config is ' + cfg.cfg_output_file)

    # output the fully-connected part into Kaldi-compatible format
    if cfg.kaldi_output_file != '':
        densenet.write_model_to_kaldi(cfg.kaldi_output_file)
        log('> ... the final Kaldi model (only FC layers) is ' + cfg.kaldi_output_file)

    # remove the tmp_DENSENET.L40_gr6.1 files (which have been generated from resuming training) 
    #os.remove(wdir + '/nnet.tmp_DENSENET.L40_gr6.1')
    #os.remove(wdir + '/training_state.tmp_DENSENET.L40_gr6.1')

#!/bin/bash

# Copyright 2017    Chia-Yu Li  University of Stuttgart       Apache 2.0
# This script trains CNN-LACEA model over the filterbank features. It  is to be run
# after run.sh. Before running this, you should already build the initial GMM
# GMM model. This script requires a GPU, and also the "pdnnlg" toolkit to train
# CNN-LACEA. The input filterbank features are with mean and variance normalization. 

# The input features and CNN-LACEA architecture follow the IBM configuration: 
# Hagen Soltau, George Saon, and Tara N. Sainath. Joint Training of Convolu-
# tional and non-Convolutional Neural Networks

working_dir=exp_pdnnlg/cnn_lacea
gmmdir=exp/tri4b

# Specify the gpu device to be used
gpu=gpu

cmd=run.pl
. cmd.sh
[ -f path.sh ] && . ./path.sh
. parse_options.sh || exit 1;

if ! nvidia-smi; then
  echo "The command nvidia-smi was not found: this probably means you don't have a GPU."
  echo "(Note: this script might still work, it would just be slower.)"
fi

# The hope here is that Theano has been installed either to python or to python2.6
pythonCMD=python
if ! python -c 'import theano;'; then
  if ! python2.6 -c 'import theano;'; then
    echo "Theano does not seem to be installed on your machine.  Not continuing."
    echo "(Note: this script might still work, it would just be slower.)"
    exit 1;
  else
    pythonCMD=python2.6
  fi
fi

mkdir -p $working_dir/log

! gmm-info $gmmdir/final.mdl >&/dev/null && \
   echo "Error getting GMM info from $gmmdir/final.mdl" && exit 1;

num_pdfs=`gmm-info $gmmdir/final.mdl | grep pdfs | awk '{print $NF}'` || exit 1;

echo =====================================================================
echo "           Data Split & Alignment & Feature Preparation            "
echo =====================================================================
# Split training data into traing and cross-validation sets for DNN
if [ ! -d data/train_tr95 ]; then
  utils/subset_data_dir_tr_cv.sh --cv-spk-percent 5 data/train_si284 data/train_tr95 data/train_cv05 || exit 1
fi
# Alignment on the training and validation data
for set in tr95 cv05; do
  if [ ! -d ${gmmdir}_ali_$set ]; then
    steps/align_fmllr.sh --nj 14   \
      data/train_$set data/lang $gmmdir ${gmmdir}_ali_$set || exit 1
  fi
done

# Generate the fbank features: 40-dimensional fbanks on each frame
echo "--num-mel-bins=40" > conf/fbank.conf
mkdir -p $working_dir/data
for set in train_tr95 train_cv05; do
  if [ ! -d $working_dir/data/$set ]; then
    cp -r data/$set $working_dir/data/$set
    ( cd $working_dir/data/$set; rm -rf {cmvn,feats}.scp split*; )
    steps/make_fbank.sh   --nj 14 $working_dir/data/$set $working_dir/_log $working_dir/_fbank || exit 1;
    steps/compute_cmvn_stats.sh $working_dir/data/$set $working_dir/_log $working_dir/_fbank || exit 1;
  fi
done

for set in dev93 eval92; do
  if [ ! -d $working_dir/data/$set ]; then
    cp -r data/test_$set $working_dir/data/$set
    ( cd $working_dir/data/$set; rm -rf {cmvn,feats}.scp split*; )
    steps/make_fbank.sh   --nj 8 $working_dir/data/$set $working_dir/_log $working_dir/_fbank || exit 1;
    steps/compute_cmvn_stats.sh $working_dir/data/$set $working_dir/_log $working_dir/_fbank || exit 1;
  fi
done

echo =====================================================================
echo "               Training and Cross-Validation Pfiles                "
echo =====================================================================
# By default, CNN-LACEA inputs include 61 frames of filterbanks
# This parameter is for the frame-skipping mechanism in order to speed up the training. e.g. 1 is no frame skipping and 2 is 2 frames skipping 
every_nth_frame=2
for set in tr95 cv05; do
  if [ ! -f $working_dir/${set}.pfile.done ]; then
    steps_pdnnlg/build_nnet_pfile.sh   --norm-vars true --add-deltas false --do-concat false \
      --splice-opts "--left-context=30 --right-context=30" \
      $working_dir/data/train_$set ${gmmdir}_ali_$set $working_dir $every_nth_frame || exit 1
    touch $working_dir/${set}.pfile.done
  fi
done

echo =====================================================================
echo "                        CNN-LACEA  Fine-tuning                           "
echo =====================================================================
# CNN-LACEA is configed in the way that it has (approximately) the same number of trainable parameters as DNN
# (e.g., the DNN in run-dnn-fbank.sh). 
if [ ! -f $working_dir/cnn_lacea.fine.done ]; then
    echo "Fine-tuning CNN"
    export CUDA_VISIBLE_DEVICES="0";
    export PYTHONPATH=$PYTHONPATH:`pwd`/pdnnlg/ ;
    export THEANO_FLAGS=mode=FAST_RUN,device=$gpu,floatX=float32 \; \
  $pythonCMD pdnnlg/cmds/run_CNN_LACEA.py --train-data "$working_dir/train_tr95.pfile.*.gz,partition=2000m,random=true,stream=true" \
                          --valid-data "$working_dir/train_cv05.pfile.*.gz,partition=600m,random=true,stream=true" \
                          --conv-nnet-spec "1x61x40:256,9x9,p1x3:256,3x4,p1x1,f" \
                          --nnet-spec "$num_pdfs" \
                          --lrate "D:0.00005:0.5:0.2,0.2:8" --momentum 0.9 \
                          --wdir $working_dir --param-output-file $working_dir/nnet.param \
                          --cfg-output-file $working_dir/nnet.cfg --kaldi-output-file $working_dir/dnn.nnet || exit 1;
  touch $working_dir/cnn_lacea.fine.done
fi

echo =====================================================================
echo "                Dump Convolution-Layer Activation                  "
echo =====================================================================
mkdir -p $working_dir/data_conv
for set in dev93; do
  if [ ! -d $working_dir/data_conv/$set ]; then
    steps_pdnnlg/make_conv_feat_lacea.sh --nj 10   \
      $working_dir/data_conv/$set $working_dir/data/$set $working_dir $working_dir/nnet.param \
      $working_dir/nnet.cfg $working_dir/_log $working_dir/_conv || exit 1;
    # Generate *fake* CMVN states here.
    steps/compute_cmvn_stats.sh --fake \
      $working_dir/data_conv/$set $working_dir/_log $working_dir/_conv || exit 1;
  fi
done
for set in eval92; do
  if [ ! -d $working_dir/data_conv/$set ]; then
    steps_pdnnlg/make_conv_feat_lacea.sh --nj 8   \
      $working_dir/data_conv/$set $working_dir/data/$set $working_dir $working_dir/nnet.param \
      $working_dir/nnet.cfg $working_dir/_log $working_dir/_conv || exit 1;
    # Generate *fake* CMVN states here.
    steps/compute_cmvn_stats.sh --fake \
      $working_dir/data_conv/$set $working_dir/_log $working_dir/_conv || exit 1;
  fi
done

echo =====================================================================
echo "                           Decoding                                "
echo =====================================================================
# In decoding, we take the convolution-layer activation as inputs and the 
# fully-connected layers as the DNN model. So we set --norm-vars, --add-deltas
# and --splice-opts accordingly.
if [ ! -f  $working_dir/decode.done ]; then
  cp $gmmdir/final.mdl $working_dir || exit 1;  # copy final.mdl for scoring
  graph_dir=$gmmdir/graph_nosp_bd_tgpr
  steps_pdnnlg/decode_cnn_lacea.sh --nj 10 --scoring-opts "--min-lmwt 7 --max-lmwt 30"   \
    --norm-vars false --add-deltas false --splice-opts "--left-context=0 --right-context=0" \
    $graph_dir $working_dir/data_conv/dev93 ${gmmdir}_ali_tr95 $working_dir/decode_nosp_bd_tgpr_dev93 || exit 1;
  steps_pdnnlg/decode_cnn_lacea.sh --nj 8 --scoring-opts "--min-lmwt 7 --max-lmwt 30"   \
    --norm-vars false --add-deltas false --splice-opts "--left-context=0 --right-context=0" \
    $graph_dir $working_dir/data_conv/eval92 ${gmmdir}_ali_tr95 $working_dir/decode_nosp_bd_tgpr_eval92 || exit 1;
  touch $working_dir/decode.done
fi

echo "Finish !!"

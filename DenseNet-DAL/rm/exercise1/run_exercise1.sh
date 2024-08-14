#!/bin/bash

# Team Lab Exercise 1

# in order to use srilm functions from the command line, set:
# export PATH=$PATH:/path/to/kaldi-trunk/tools/srilm/bin/i686-m64

# TODO add current working directory
dir=`pwd`

# path to a previous language model
lm=/mount/projekte/slu/Lectures/SS2018/teamlab/ASR/exercise1/lm_tgpr.arpa.gz

# Step 1

# Train a new LM using SRILM from arbitrary, pre-normalized text
# Interpolate it with a language model that already has been trained on the WSJ corpus
# Test on WSJ development data

# TODO take a look on the script the understand all the arguments
local/train_lms_srilm_interp.sh data/lm/data/normalized_text.txt ../s5/data/lang_nosp/words.txt ../s5/data/test_dev93/text $lm data/lm/ 0.7 >> run.log

# the last argument is the value for lambda
# this is the weight that the main LM, in this case $lm, is given
# e.g. 0.5 means that both LMs will be weighted equally


# Step 2

# Test the new interpolated language model on WSJ test data
# using the triphone acoustic model trained on WSJ training data
# Refer to the script for more details and hard-coded paths!

# TODO update the current path in test-lm.sh and the path in arpa2G.sh 

# TODO insert path to best lm here:
local/test-lm.sh test_new <path to the best lm> <order> >> run.log

# compare using original lm: 
local/test-lm.sh test_orig $lm 3 >> run.log



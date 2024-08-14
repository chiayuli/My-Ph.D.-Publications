The code was originally written by [@kaldipdnn](https://github.com/yajiemiao/kaldipdnn) and modified by [@Chia-Yu Li](https://github.com/chiayuli).

# Citations
```
@InProceedings{Li2019_55,
author = {Chia Yu Li and Ngoc Thang Vu},
booktitle = {Studientexte zur Sprachkommunikation: Elektronische Sprachsignalverarbeitung 2019},
title = {Investigation of densely connected convolutional networks with domain adversarial learning for noise robust speech recognition},
year = {2019},
editor = {Peter Birkholz and Simon Stone},
month = mar,
pages = {9--16},
publisher = {TUDpress, Dresden}
}
```

# Datasets
* [@AURORA4](http://aurora.hsnr.de/aurora-4.html)
* self-developed noise corrupted Resource Management 2.0 using [@our_scripts](https://github.com/chiayuli/noise-data-preparation) and [@idiap_acoustic_simulator](https://github.com/idiap/acoustic-simulator)

# Toolkits
Kaldi and install_pfile_utils.sh

# run scripts
* The recipes and tools for training DNN/CNN/CNN-LACE/DENSENET/DENSENET-DAL based acoustic models are in the
  directories of aurora/ and rm/.
* In pdnnlg, you could find lots of different types of network architectures and using different hyperparameters showing on the filename. (e.g., pdnnlg_ad/cmds/run_DENSENET_L61_12_test2_theta0_5_test3_bn.py is a 61-layers DenseNet with theta 0.5 and batchnorm)
* An example of command to start model training 
```
export CUDA_VISIBLE_DEVICES=1; THEANO_FLAGS='device=cuda' python pdnnlg_ad/cmds/run_DENSENET_L61_12_test2_theta0_4_test5_bn.py \
        --train-data "$working_dir/train_multi_tr95.pfile.1.gz,partition=8000m,random=true,stream=true" \
        --valid-data "$working_dir/train_multi_cv05.pfile.1.gz,partition=600m,random=true,stream=true" \
        --conv-nnet-spec "3x11x40:256,9x9,p1x3:256,3x4,p1x1,f" --nnet-spec "$num_pdfs" --lrate "D:0.01:0.5:0.2,0.2:6" \
        --momentum 0.9 --wdir $working_dir --param-output-file $working_dir/nnet.param --cfg-output-file $working_dir/nnet.cfg \
        --kaldi-output-file $working_dir/dnn.nnet

```
* An example command to do decoding
```
./densenet_L61_12_test2_theta-decode.sh
```

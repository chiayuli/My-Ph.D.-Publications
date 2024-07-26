The code was originally written by [@kaldipdnn](https://github.com/yajiemiao/kaldipdnn) and modified by [@Chia-Yu Li](https://github.com/chiayuli).

# Citations
```
@INPROCEEDINGS{8578047,
  author={Li, Chia Yu and Vu, Ngoc Thang},
  booktitle={Speech Communication; 13th ITG-Symposium}, 
  title={Densely Connected Convolutional Networks for Speech Recognition}, 
  year={2018},
  volume={},
  number={},
  pages={1-5},
  keywords={},
  doi={}}
```

# Datasets
* SWBD (LDC97S62)
* WSJ (LDC97S62 and LDC94S13A)

# Toolkits
Kaldi and install_pfile_utils.sh

# run scripts
* For SWBD task, the recipes and tools for training DNN/CNN/CNN-LACE/DENSENET based acoustic models are in the swbd/
* For WSJ task, those are in wsj/
* gmm sat model (tri4) and baseline (DNN, CNN, CNN-LACEA) and DENSENET acoustic model training and evaluation
```
./run.sh
./pdnnlg/run-dnn.sh
./pdnnlg/run-dnn-fbank.sh
./pdnnlg/run-cnn.sh
./pdnnlg/run-cnn-lacea.sh
./pdnnlg/run-densenet.sh
```

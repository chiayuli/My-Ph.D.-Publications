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
publisher = {TUDpress, Dresden},
abstract = {We investigate densely connected convolutional networks (DenseNets)and their extension with domain adversarial training for noise robust speech recog-nition. DenseNets are very deep, compact convolutional neural networks whichhave demonstrated incredible improvements over the state-of-the-art results incomputer vision. Our experimental results reveal that DenseNets are more robustagainst noise than other neural network based models such as deep feed forwardneural networks and convolutional neural networks. Moreover, domain adversariallearning can further improve the robustness of DenseNets against both, known andunknown noise conditions.},
isbn = {978-3-959081-57-3},
issn = {0940-6832},
keywords = {Spracherkennung und -wahrnehmung},
url = {https://www.essv.de/pdf/2019_9_16.pdf},
}
```

# Datasets
* AURORA4 (http://aurora.hsnr.de/aurora-4.html)
* self-developed noise corrupted Resource Management 2.0 using our [@scripts](https://github.com/chiayuli/noise-data-preparation) and [@idiap_acoustic_simulator](https://github.com/idiap/acoustic-simulator)

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

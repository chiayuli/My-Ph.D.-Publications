The code was originally written by [@Espnet-group](https://github.com/espnet/espnet) and modified by [@Chia-Yu Li](https://github.com/chiayuli).

# Citations
```
@INPROCEEDINGS{9037688,
  author={Li, Chia-Yu and Vu, Ngoc Thang},
  booktitle={2019 International Conference on Asian Language Processing (IALP)}, 
  title={Integrating Knowledge in End-to-End Automatic Speech Recognition for Mandarin-English Code-Switching}, 
  year={2019},
  volume={},
  number={},
  pages={160-165},
  keywords={Hidden Markov models;Speech recognition;Switches;Dictionaries;Acoustics;Decoding;Computational modeling;end-to-end speech recognition;Mandarin-English Code-Switching speech;language model integration},
  doi={10.1109/IALP48816.2019.9037688}}
```
# Datasets
* SEAME: Mandarin-English Code-Switching in South-East Asia (LDC2015S04) for training baseline ASR and CycleGAN
* [@Aishell-1](https://www.openslr.org/33/), [@SMS](https://scholarbank.nus.edu.sg/handle/10635/137343) for training Sequence-to-sequence (S2S) model (baseline) and CycleGAN.
* [@THCHS-30](https://www.openslr.org/18/): A Free Chinese Speech Corpus Released by CSLT@Tsinghua University
* [@ST-CMDS](https://www.openslr.org/38/): A free Chinese Mandarin corpus
* [@Librispeech](https://www.openslr.org/12): Large-scale (1000 hours) corpus of read English speech
* [@Common Voice](https://commonvoice.mozilla.org/en/datasets) (old version, 425 hours)
* [@TedXSingapore](https://www.ted.com/tedx/events/56510)

# ASR toolkit: espnet
* Please follow the steps in https://github.com/espnet/espnet to install espnet
  
# Run scripts
* use old utils to compute the MER
```
  cp <your-espnet>/src/utils/score_sclite.sh <your-espnet>/src/utils/score_sclite.sh.bk
  cp <your-espnet>/src/utils/json2trn.py <your-espnet>/src/utils/json2trn.py.bk
  cp utils/{score_sclite.sh,json2trn.py} <your-espnet>/src/utils/
```
* preprocessing: conf/specaug.ymal or conf/no_preprocess.ymal
* model type: conf/train_rnn.ymal (VGGBLSTM) or conf/train.ymal (Transformer) 
  
```
export CUDA_VISIBLE_DEVICES=0,1,2; nohup ./run.sh --ngpu 3 --skip-lm_training false --preprocess_config <preprocess-config-ymal> --train_config <model-config-ymal> >> run.log&

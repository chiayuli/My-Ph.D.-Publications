The code was originally written by [@Espnet-group](https://github.com/espnet/espnet) and modified by [@Chia-Yu Li](https://github.com/chiayuli).

# Citations
```
@INPROCEEDINGS{10022448,
  author={Li, Chia-Yu and Vu, Ngoc Thang},
  booktitle={2022 IEEE Spoken Language Technology Workshop (SLT)}, 
  title={Improving Semi-Supervised End-To-End Automatic Speech Recognition Using Cyclegan and Inter-Domain Losses}, 
  year={2023},
  volume={},
  number={},
  pages={822-829},
  keywords={Error analysis;Conferences;Feature extraction;Automatic speech recognition;speech recognition;End-to-end;semi-supervised training;CycleGAN},
  doi={10.1109/SLT54892.2023.10022448}}
```
# Datasets
* WSJ (LDC93S6A, LDC94S13A)
* [@Librispeech](https://www.openslr.org/12): Large-scale (1000 hours) corpus of read English speech

# Installation and initialization
* copy espnet/espnet to you working directory and run installation following the README.md in [@Espnet](https://github.com/espnet/espnet)
* create a folder naming "semi" under espnet/espnet/egs/wsj

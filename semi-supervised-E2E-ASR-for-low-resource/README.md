The code was originally written by [@Espnet-group](https://github.com/espnet/espnet) and modified by [@Chia-Yu Li](https://github.com/chiayuli).

# Citations
```
@INPROCEEDINGS{cNST,
  author={Li, Chia-Yu and Vu, Ngoc Thang},
  booktitle={the 3rd Annual Meeting of the Special Interest Group on Under-resourced Languages (SIGUL)}, 
  title={Improving noisy student training for low-resource languages in E2E ASR using CycleGAN-inter-domain losses.}, 
  year={2024}}
```
# Datasets
* Voxforge
* [@Common Voice](https://commonvoice.mozilla.org/en/datasets) (old version, 425 hours)

# Experiment
* please copy all the files under espnet to your espnet directory
* please go to the egs/commonvoice/asr1/local or egs/voxforge/asr1/local, and copy local/process_sentence.sh to it
* for Voxforge, the run script is run.vx.final.sh
* for common voice, the run script is run.cv.final.sh, and the data split are in data/{el,fi,hu}

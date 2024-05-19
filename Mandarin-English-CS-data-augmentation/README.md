The code was initially written by [@Chia-Yu Li](https://github.com/chiayuli).

# Citations
```
@INPROCEEDINGS{inproceedings,
author = {Li, Chia-Yu and Vu, Thang},
year = {2020},
month = {10},
pages = {1057-1061},
title = {Improving Code-Switching Language Modeling with Artificially Generated Texts Using Cycle-Consistent Adversarial Networks},
doi = {10.21437/Interspeech.2020-2177}
}

```
# Datasets
* SEAME: Mandarin-English Code-Switching in South-East Asia (LDC2015S04) for training baseline ASR and CycleGAN
* [@Aishell-1](https://www.openslr.org/33/), [@SMS](https://scholarbank.nus.edu.sg/handle/10635/137343) for training Sequence-to-sequence (S2S) model (baseline) and CycleGAN.
* self-developed translationed based Mandarin-English CS text using google translator API

# ASR toolkit
* Kaldi (https://github.com/kaldi-asr/kaldi)

# Executing run.sh script
Let A represents the Mandarin text's domain, and B represents the domain of the Mandarin-English CS text. 
* We pretrained the two generators (G:A->B and F:B->A) and discriminators with the transilation based synthetic Mandarin-English CS text.
```
expdir=exp/AISSMS_40K_2/
for s in "A2B" "B2A" "Discriminator"
do
  export CUDA_VISIBLE_DEVICES=2; nohup python -u pretrain.py --model $s --exp_dir $expdir > run.log.pretrain${s} &
done
```
* Train CycleGAN model with SEAME data and self-developed CS data
```
expdir=exp/AISSMS_40K_2/
export CUDA_VISIBLE_DEVICES=2; nohup python -u cycleGAN.py --lam_A2B2A 0.1 --lam_B2A2B 0.6 --exp_dir $expdir > run.log.cycleGAN &
expdir=exp/CCG_40K_B2A2B0.4A2B2A0
export CUDA_VISIBLE_DEVICES=2; nohup python -u cycleGAN_resume.py --resume exp/CCG_40K_B2A2B0.4A2B2A0/CCG4_bs10_netA2B.pt --start_epoch 4  --lam_A2B2A 0 --lam_B2A2B 0.4 --exp_dir $expdir > run.B2A2B0.4A2B2A0.log &
```
* /mount/arbeitsdaten/asr/licu/Experiments/SE_Work/CycleGAN
* 

The code was written by [Chia-Yu Li](https://github.com/chiayuli).

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
* self-developed translationed based Mandarin-English CS text using [Google translator API](https://cloud.google.com/translate/docs/reference/rest)

# ASR toolkit
* Kaldi (https://github.com/kaldi-asr/kaldi)

# Executing run.sh script
Let A represents the Mandarin text's domain, and B represents the domain of the Mandarin-English CS text. 
* We pretrained the two generators (G:A->B and F:B->A) and discriminators with the translation based synthetic Mandarin-English CS text.
```
expdir=exp/AISSMS_40K_2/
for s in "A2B" "B2A" "Discriminator"
do
  export CUDA_VISIBLE_DEVICES=2;
  nohup python -u pretrain.py --model $s --exp_dir $expdir > run.log.pretrain${s} &
done
```
* Train CycleGAN model with SEAME data and self-developed CS data
```
expdir=exp/AISSMS_40K_2/
export CUDA_VISIBLE_DEVICES=2;
nohup python -u cycleGAN.py --lam_A2B2A 0.1 --lam_B2A2B 0.6 --exp_dir $expdir > run.log.cycleGAN &
```
* Concatenate all the output of generators given different Mandarin input text and normalize them before language model training
```
# cat all the generated text
bs=10
i=9
outdir=exp/AISSMS_40K_B2A2B0.6_A2B2A0.3
mkdir -p $outdir
cat $expdir/CCG_EP${i}_bs${bs}_test_G $expdir/CCG_EP${i}_bs${bs}_train_allG $expdir/CCG_EP${i}_bs${bs}_valid_allG $expdir/CCG_EP${i}_bs${bs}_train_PKUSMS $expdir/CCG_EP${i}_bs${bs}_valid_PKUSMS > $outdir/CCG_EP${i}_bs${bs}_148853

# ./normalize_text.sh <text> <expdir>
tag=B2A2B0.6_A2B2A0.3
[ ! -d exp/normalized_text/${tag} ] && mkdir -p exp/normalized_text/${tag}
./normalize_text.sh $outdir/CCG_EP${i}_bs${bs}_148853 exp/normalized_text/${tag}
```
* Combine the generated text with SEAME train text
* Train a language model (LM) with mixed data, evaluate LM ppl on SEAME test set and run inference on SEAME test set using our best [Mandarin-English CS ASR](https://github.com/chiayuli/My-Ph.D.-Publications/tree/main/E2E-ASR-for-Mandarin-English-CS-speech) with LM
```
cd $E2E/egs/cs/asr5_0/
tag=B2A2B0.6_A2B2A0.3
bs=10
i=9
mkdir -p data/seametrain_${tag}_EP${i} 
cd data/seametrain_${tag}_EP${i}
ln -s $CS/exp/normalized_text/${tag}/CCG_EP${i}_bs${bs}_148853_norm2 .
cut -d' ' -f2- data/seame-train/text | cat - CCG_EP${i}_bs${bs}_148853_norm2 > train.txt
cut -d' ' -f2- data/seame-dev/text > valid.txt
cut -d' ' -f2- data/seame-eval/text > test.txt

export CUDA_VISIBLE_DEVICES=2; nohup ./run.CEF3.CCG.sh --stage 3 --ngpu 1 >> run.log.CCG5.test&
```

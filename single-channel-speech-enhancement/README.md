The code was initially written by [@Jun-Yan Zhu](https://github.com/junyanz), [@Taesung Park](https://github.com/taesungp), and [@Tongzhou Wang](https://github.com/SsnL) and modified by [@Chia-Yu Li](https://github.com/chiayuli).

# Citations
```
@INPROCEEDINGS{9688310,
  author={Li, Chia-Yu and Vu, Ngoc Thang},
  booktitle={2021 IEEE Automatic Speech Recognition and Understanding Workshop (ASRU)}, 
  title={Improving Speech Recognition on Noisy Speech via Speech Enhancement with Multi-Discriminators CycleGAN}, 
  year={2021},
  volume={},
  number={},
  pages={830-836},
  keywords={Training;Conferences;Training data;Speech enhancement;Generators;Noise measurement;Spectrogram;Speech recognition;noisy speech;Cycle-GAN;Speech Enhancement},
  doi={10.1109/ASRU51503.2021.9688310}}

```
# Datasets
* CHiME3 (LDC2017S24) for training multi-discriminators CycleGAN
* WSJ (LDC93S6A, LDC94S13A) for training ASR

# ASR toolkit
* Kaldi (https://github.com/kaldi-asr/kaldi)

# Executing scripts
* Prepare parallel data (clean, noise) for training CycleGAN.
```
./prepare_trainSet.sh
```
* Training multi-discriminator CycleGAN with the paralleled data
```
# Command for training CycleGAN: EXP-1DA/cmd.txt
# Command for training 2-discriminator CycleGAN: EXP-2DA/cmd.txt
# Command for training 3-discriminator CycleGAN: EXP-3DA/cmd.txt
./train.py --dataroot ./datasets/chime3_v12 --name cyclegan_chime3_v12_resnet_6blocks_v13 --model cycle_gan --batch_size 256 --gpu_ids 1,2,3 --ngf 256 --netG resnet_6blocks --netD n_layers --n_layers_D 6 --lambda_A 20 

# Write features to kaldi format
./test_real_v12_res6b_v13.py --dataroot ./datasets/chime3_v5 --name cyclegan_chime3_v12_resnet_6blocks_v13  --model cycle_gan --gpu_ids 2 --ngf 256 --netG resnet_6blocks --netD n_layers --n_layers_D 6 >> run.log.v12.resnet_6blocks_v13.new &

# Plot Mel-spectrogram
plotSpec.py
```


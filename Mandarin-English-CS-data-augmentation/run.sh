source ../venv/bin/activate
# prepare translation based synthetic CS text
expdir=exp/AISSMS_40K_2/
for s in "A2B" "B2A" "Discriminator"
do
  export CUDA_VISIBLE_DEVICES=2; nohup python -u pretrain.py --model $s --exp_dir $expdir > run.log.pretrain${s} &
done

# train CycleGAN models
expdir=exp/AISSMS_40K_2/
export CUDA_VISIBLE_DEVICES=2; nohup python -u cycleGAN.py --lam_A2B2A 0.1 --lam_B2A2B 0.6 --exp_dir $expdir > run.log.cycleGAN &
expdir=exp/CCG_40K_B2A2B0.4A2B2A0
export CUDA_VISIBLE_DEVICES=2; nohup python -u models_resume.py --resume exp/CCG_40K_B2A2B0.4A2B2A0/CCG4_bs10_netA2B.pt --start_epoch 4  --lam_A2B2A 0 --lam_B2A2B 0.4 --exp_dir $expdir > run.B2A2B0.4A2B2A0.log &

# cat all the generated text
bs=10
i=9
outdir=exp/AISSMS_40K_B2A2B0.6_A2B2A0.3
mkdir -p $outdir
cat $expdir/CCG_EP${i}_bs${bs}_test_G $expdir/CCG_EP${i}_bs${bs}_train_allG $expdir/CCG_EP${i}_bs${bs}_valid_allG $expdir/CCG_EP${i}_bs${bs}_train_PKUSMS $expdir/CCG_EP${i}_bs${bs}_valid_PKUSMS > $outdir/CCG_EP${i}_bs${bs}_148853

# normalize the predictions
# utils/normalize_text.sh <text> <expdir>
tag=B2A2B0.6_A2B2A0.3
[ ! -d exp/normalized_text/${tag} ] && mkdir -p exp/normalized_text/${tag}
utils/normalize_text.sh $outdir/CCG_EP${i}_bs${bs}_148853 exp/normalized_text/${tag}

# train rnnlm with normalized synthetic data
cd $E2E/egs/cs/asr5_0/
tag=B2A2B0.6_A2B2A0.3
bs=10
i=9
mkdir -p data/seametrain_${tag}_EP${i} 
cd data/seametrain_${tag}_EP${i}
ln -s $CS/exp/normalized_text/${tag}/CCG_EP${i}_bs${bs}_148853_norm2 .
cut -d' ' -f2- ../train/text | cat - CCG_EP${i}_bs${bs}_148853_norm2 > train.txt
cut -d' ' -f2- ../dev/text > valid.txt
cut -d' ' -f2- ../eval/text > test.txt
cd ..

export CUDA_VISIBLE_DEVICES=2; nohup ./run.CEF3.CCG.sh --stage 3 --ngpu 1 >> run.log.CCG5.test&

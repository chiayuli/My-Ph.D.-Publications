modeldir=$1
testdir=$2
epoch=$3
tag=$4 #"dt05_real_noisy_exp03_v1_fbank"
gpu1=$5
gpu2=$6

if [ $# -lt 5 ];
then
  echo "$0 <modeldir> <testdir> <epoch> <tag> <gpu1> <gpu2>"
  exit
fi
## network settings
featDim=40
#ngf=256
model=cycle_gan
#G=resnet_6blocks
#D=n_layers
#nlayersD=5
source /mount/arbeitsdaten/asr/licu/SE_Work/miniconda3/bin/activate
PYTHON=/mount/arbeitsdaten/asr/licu/SE_Work/miniconda3/bin/python

[ ! -d output_feat/logs ] && mkdir -p output_feat/logs
outdir=output_feat/${tag} && mkdir -p $outdir
cp -r $testdir/testA/* ${outdir}/
rm ${outdir}/split4/*/feats.scp
rm ${outdir}/split4/*/cmvn.scp

for i in {1..2};
do
    $PYTHON generate_feats.py --dataroot ${testdir}/testA/split4/${i} \
        --epoch ${epoch} --outdir ${outdir}/split4/${i}/ --phase "test" \
        --name ${modeldir} --model ${model} --featDim ${featDim} \
        --gpu_ids ${gpu1} > output_feat/logs/run.log.${tag}.output.${i} &
done

for i in {3..4};
do
    $PYTHON generate_feats.py --dataroot ${testdir}/testA/split4/${i} \
        --epoch ${epoch} --outdir ${outdir}/split4/${i}/ --phase "test" \
        --name ${modeldir} --model ${model} --featDim ${featDim} \
        --gpu_ids ${gpu2} > output_feat/logs/run.log.${tag}.output.${i} &
done




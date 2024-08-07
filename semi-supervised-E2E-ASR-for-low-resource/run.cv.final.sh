#!/bin/bash

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

# general configuration
backend=pytorch
stage=8      # start from -1 if you need to start from data download
stop_stage=100
ngpu=1         # number of gpus ("0" uses cpu, otherwise use gpu)
debugmode=1
dumpdir=dump   # directory to dump full features
N=0            # number of minibatches to be used (mainly for debugging). "0" uses all minibatches.
resume=        # Resume the training from snapshot

# feature configuration
do_delta=false

# dictoinary configuratoin
nbpe=

# config files
init_preprocess_config=conf/specaug.yaml  # use conf/specaug.yaml for data augmentation
semi_preprocess_config=conf/specaug.yaml  # use conf/specaug.yaml for data augmentation
lm_config=conf/lm.yaml
init_train_config=conf/tuning/train_rnn_small_bs5.yaml
semi_train_config=conf/tuning/train_rnn_small_bs5.yaml
train_config=conf/tuning/train_rnn_small_bs5.yaml
decode_config=conf/tuning/decode.yaml
nbpe=

# LM related
lmtag=
lm_resume=
skip_lm_training=false

# semi-supervised loss
seed=1
unsupervised_loss=mmd
sup_loss_ratio=0.5
sup_data_ratio=1.0
use_batchnorm=False
weight_decay=0.0
learning_rate=1.0
init_learning_rate=
sup_ratio_decay=1 # 0: not decay, 1: alpha from 0.9 -> 0.5 when epoch > 2, 4, 5, 6
auto_st_tuning=min # non, min, max, avg, med
add_speech=true
# decoding parameter
recog_model=model.acc.best # set a model to be used for decoding: 'model.acc.best' or 'model.loss.best'

# model average realted (only for transformer)
n_average=10                 # the number of ASR models to be averaged
use_valbest_average=false    # if true, the validation `n_average`-best ASR models will be averaged.
                             # if false, the last `n_average` ASR models will be averaged.
debug=true

# common voice data 
lang=hu # fi, el 

# exp tag
volume="" # volume of text data
. utils/parse_options.sh || exit 1;

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

# automatic hyperparameters tuning proposed in paper
alpha_ver=3
if [[ "$sup_ratio_decay" -eq "0" ]]; then
    echo "supervised ratio decay is disabled"
    alpha_ver=2
elif [[ "$sup_ratio_decay" -eq "1" ]]; then
    echo "supervised ratio decay is enabled (0.9 -> 0.8 after epoch > 4)"
    alpha_ver=3
elif [[ "$sup_ratio_decay" -eq "2" ]]; then
    echo "supervised ratio decay is enabled (0.9 -> 0.5 after epoch > 2,4,5,6)"
    alpha_ver=6
fi

beta_ver=2
if [[ "$auto_st_tuning" == "non" ]]; then
    echo "automatic speech-to-text ratio tuning disabled"
    beta_ver=1
elif [[ "$auto_st_tuning" == "min" ]]; then
    echo "automatic speech-to-text ratio tuning: min()"
    beta_ver=2
elif [[ "$auto_st_tuning" == "max" ]]; then
    echo "automatic speech-to-text ratio tuning: max()"
    beta_ver=3
elif [[ "$auto_st_tuning" == "avg" ]]; then
    echo "automatic speech-to-text ratio tuning: avg()"
    beta_ver=4
elif [[ "$auto_st_tuning" == "med" ]]; then
    echo "automatic speech-to-text ratio tuning: med()"
    beta_ver=5
fi

ver=${alpha_ver}${beta_ver}


# training config for different languages
if [ ${lang} == "hu" ]; then
  volume="100K"
  init_learning_rate="0.5"
  semi_learning_rate="0.5"
  learning_rate="1.0"
  init_train_config=conf/tuning/train_rnn_small3_bs10_init.yaml
  semi_train_config=conf/tuning/train_rnn_small_bs10.yaml
  train_config=conf/tuning/train_rnn_small_bs5.yaml
  lm_train_set=train_${lang}_init_and_fake_${volume}
elif [ ${lang} == "el" ]; then
  volume="30KWEB2015"
  init_learning_rate="0.5"
  semi_learning_rate="1.0"
  learning_rate="0.5"
  init_train_config=conf/tuning/train_rnn_small3_bs10_init.yaml
  semi_train_config=conf/tuning/train_rnn_small_bs10.yaml
  train_config=conf/tuning/train_rnn_small_bs10.yaml
  lm_train_set=train_${lang}_init_and_us
elif [ ${lang} == "fi" ]; then
  volume="300KWEB"
  add_speech=false
  init_learning_rate="0.5"
  semi_learning_rate="0.5"
  learning_rate="1.0"
  init_train_config=conf/tuning/train_rnn_small_bs10_init.yaml
  semi_train_config=conf/tuning/train_rnn_small_bs10.yaml
  train_config=conf/tuning/train_rnn_small_bs10.yaml
  lm_train_set=train_${lang}_init_and_fake_${volume}
fi

if ${add_speech}; then
    unpair_set=train_${lang}_fake_${volume}   # unpair text and train_lang_us speech
else
    unpair_set=train_${lang}_fake_${volume}_noAddSpeech   # only unpair text and train_init speech
fi

# bpemode (unigram or bpe)
nbpe=150
bpemode=unigram

lm_train_dev=dev_${lang}
lm_train_test=test_${lang}_big
train_set=train_${lang}_init
unlabel_set=train_${lang}_us
train_dev=dev_${lang}
recog_set="test_${lang}_big"

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    ### Task dependent. You have to make data the following preparation part by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 0: Data Preparation"
    if [ ! -d data/train_${lang} ]; then ## need to run oracle first
        echo "please execute run.sh first"
    fi
    # remove train_init data from train set
    utils/copy_data_dir.sh data/train_${lang} data/${unpair_set}
    utils/filter_scp.pl --exclude data/${train_set}/wav.scp data/${unpair_set}/wav.scp  > data/${unpair_set}/temp_wav.scp
    mv data/${unpair_set}/temp_wav.scp data/${unpair_set}/wav.scp
    utils/fix_data_dir.sh data/${unpair_set}

   # prepare external text
   # please change download the text first, and then run this script
   local/process_sentence.sh
fi

feat_tr_dir=${dumpdir}/${train_set}/delta${do_delta}; mkdir -p ${feat_tr_dir}
feat_us_dir=${dumpdir}/${unpair_set}/delta${do_delta}; mkdir -p ${feat_us_dir}
feat_dt_dir=${dumpdir}/${train_dev}/delta${do_delta}; mkdir -p ${feat_dt_dir}
feat_unlabel_dir=${dumpdir}/${unlabel_set}/delta${do_delta}; mkdir -p ${feat_unlabel_dir}
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "stage 1: Dump Feature"
    fbankdir=fbank
    for set in "${train_set}" "${unpair_set}" "${train_dev}" "${recog_set}" "${unlabel_set}"
    do
        if [ ! -f ${fbankdir}/$lang/raw_fbank_pitch_${set}.1.ark ]; then
            echo "make features for ${set}"
            nj=`wc -l data/${set}/spk2utt | cut -d" " -f1`
            [ "$nj" -gt "48" ] && nj=40
            steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj $nj --write_utt2num_frames true \
                data/${set} exp/make_fbank/${set} ${fbankdir}
            utils/fix_data_dir.sh data/${set}
        fi
    done

    dump.sh --cmd "$train_cmd" --nj 4 --do_delta ${do_delta} \
        data/${train_set}/feats.scp data/train_${lang}/cmvn.ark exp/dump_feats/${train_set} \
        ${feat_tr_dir}
    dump.sh --cmd "$train_cmd" --nj $nj --do_delta ${do_delta} \
        data/${unpair_set}/feats.scp data/train_${lang}/cmvn.ark exp/dump_feats/${unpair_set} \
        ${feat_us_dir}
    dump.sh --cmd "$train_cmd" --nj 4 --do_delta ${do_delta} \
        data/${train_dev}/feats.scp data/train_${lang}/cmvn.ark exp/dump_feats/${train_dev} \
        ${feat_dt_dir}
    dump.sh --cmd "$train_cmd" --nj 4 --do_delta ${do_delta} \
        data/${unlabel_set}/feats.scp data/train_${lang}/cmvn.ark exp/dump_feats/${unlabel_set} \
        ${feat_unlabel_dir}
    for rtask in ${recog_set}; do
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}; mkdir -p ${feat_recog_dir}
        dump.sh --cmd "$train_cmd" --nj 4 --do_delta ${do_delta} \
            data/${rtask}/feats.scp data/train_${lang}/cmvn.ark exp/dump_feats/recog/${rtask} \
            ${feat_recog_dir}
    done
fi

dict=data/${lang}_lang_char/train_${lang}_${bpemode}${nbpe}_units.txt
bpemodel=data/${lang}_lang_char/train_${lang}_${bpemode}${nbpe}
echo "dictionary: ${dict}"
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    ### Task dependent. You have to check non-linguistic symbols used in the corpus.
    echo "stage 2: Dictionary and Json Data Preparation"
    mkdir -p data/${lang}_lang_char/
    echo "make a dictionary"
    echo "<unk> 1" > ${dict} # <unk> must be 1, 0 will be used for "blank" in CTC
    cut -f 2- -d" " data/${train_set}/text > data/${lang}_lang_char/input.txt
    spm_train --input=data/${lang}_lang_char/input.txt --vocab_size=${nbpe} --model_type=${bpemode} --model_prefix=${bpemodel} --input_sentence_size=100000000
    spm_encode --model=${bpemodel}.model --output_format=piece < data/${lang}_lang_char/input.txt | tr ' ' '\n' | sort | uniq | awk '{print $0 " " NR+1}' >> ${dict}
    wc -l ${dict}

    echo "make json files"
    data2json.sh --feat ${feat_tr_dir}/feats.scp --bpecode ${bpemodel}.model \
        data/${train_set} ${dict} > ${feat_tr_dir}/data_${bpemode}${nbpe}.json
    data2json.sh --feat ${feat_us_dir}/feats.scp --bpecode ${bpemodel}.model \
        data/${unpair_set} ${dict} > ${feat_us_dir}/data_${bpemode}${nbpe}.json
    data2json.sh --feat ${feat_dt_dir}/feats.scp --bpecode ${bpemodel}.model \
        data/${train_dev} ${dict} > ${feat_dt_dir}/data_${bpemode}${nbpe}.json
    data2json.sh --feat ${feat_unlabel_dir}/feats.scp --bpecode ${bpemodel}.model \
        data/${unlabel_set} ${dict} > ${feat_unlabel_dir}/data_${bpemode}${nbpe}.json
    for rtask in ${recog_set}; do
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}
        data2json.sh --feat ${feat_recog_dir}/feats.scp --bpecode ${bpemodel}.model \
            data/${rtask} ${dict} > ${feat_recog_dir}/data_${bpemode}${nbpe}.json
    done
fi

if [ -z ${lmtag} ]; then
    lmtag=$(basename ${lm_config%.*})_charlm
fi

lmexpname=rnnlm_${lm_train_set}_${backend}_${lmtag}_${bpemode}${nbpe}
lmexpdir=exp/${lang}/${lmexpname}
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ] && ! ${skip_lm_training}; then
    echo "stage 3: LM Preparation"
    mkdir -p ${lmexpdir}
    lmdatadir=data/local/lm_${lm_train_set}
    lmdict=${dict}
    mkdir -p ${lmdatadir}
    cut -f 2- -d" " data/${lm_train_set}/text | spm_encode --model=${bpemodel}.model --output_format=piece > ${lmdatadir}/train.txt
    cut -f 2- -d" " data/${lm_train_dev}/text | spm_encode --model=${bpemodel}.model --output_format=piece > ${lmdatadir}/valid.txt
    cut -f 2- -d" " data/${lm_train_test}/text | spm_encode --model=${bpemodel}.model --output_format=piece > ${lmdatadir}/test.txt

    ${cuda_cmd} --gpu ${ngpu} ${lmexpdir}/train.log \
        lm_train.py \
        --config ${lm_config} \
        --ngpu ${ngpu} \
        --backend ${backend} \
        --verbose 1 \
        --outdir ${lmexpdir} \
        --tensorboard-dir tensorboard/${lmexpname} \
        --train-label ${lmdatadir}/train.txt \
        --valid-label ${lmdatadir}/valid.txt \
        --test-label ${lmdatadir}/test.txt \
        --resume ${lm_resume} \
        --dict ${lmdict}
fi

if ${debug}; then
  expname=Step1_${train_set}_${backend}_$(basename ${init_train_config%.*})_$(basename ${init_preprocess_config%.*})_lr${init_learning_rate/./}
else
  expname=Step1_${train_set}_${backend}_$(basename ${init_train_config%.*})_$(basename ${init_preprocess_config%.*})
fi
expdir=exp/${lang}/${expname}/

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "stage 4: Init Network Training (M_0)"
    mkdir -p ${expdir}
    ${cuda_cmd} --gpu ${ngpu} ${expdir}/train.log \
        asr_train.py \
        --config ${init_train_config} \
        --preprocess-conf ${init_preprocess_config} \
        --ngpu ${ngpu} \
        --backend ${backend} \
        --outdir ${expdir}/results \
        --tensorboard-dir tensorboard/${expname} \
        --debugmode ${debugmode} \
        --dict ${dict} \
        --debugdir ${expdir} \
        --minibatches ${N} \
        --verbose 0 \
        --resume ${resume} \
        --train-json ${feat_tr_dir}/data_${bpemode}${nbpe}.json \
        --valid-json ${feat_dt_dir}/data_${bpemode}${nbpe}.json
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    echo "stage 5: Decoding"
    nj=16
    pids=() # initialize pids
    for rtask in "${unlabel_set}" "${recog_set}" ; do
    (
        recog_opts=
        if ${skip_lm_training}; then
           lmtag="nolm"
        else
           lmtag="lm"
           recog_opts="--rnnlm ${lmexpdir}/rnnlm.model.best"
        fi
        decode_dir=decode_${rtask}_$(basename ${decode_config%.*})_${lmtag}
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}

        # split data
        splitjson.py --parts ${nj} ${feat_recog_dir}/data_${bpemode}${nbpe}.json

        #### use CPU for decoding
        ngpu=0

        ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
            asr_recog.py \
            --config ${decode_config} \
            --ngpu ${ngpu} \
            --backend ${backend} \
            --debugmode ${debugmode} \
            --recog-json ${feat_recog_dir}/split${nj}utt/data_${bpemode}${nbpe}.JOB.json \
            --result-label ${expdir}/${decode_dir}/data.JOB.json \
            --model ${expdir}/results/${recog_model} \
            ${recog_opts}

        score_sclite.sh --bpe ${nbpe} --bpemodel ${bpemodel}.model --wer true ${expdir}/${decode_dir} ${dict}

    )&
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
    echo "Finished"
fi

echo "${dict}"
init_model=${expdir}/results/model.acc.best

if ${debug}; then
  expname=Step2_semi_${unpair_set}_${backend}_$(basename ${semi_train_config%.*})_$(basename ${semi_preprocess_config%.*})_initlr${init_learning_rate/./}_lr${semi_learning_rate/./}
else
  expname=Step2_semi_${unpair_set}_${backend}_$(basename ${semi_train_config%.*})_$(basename ${semi_preprocess_config%.*})
fi
expdir=exp/${lang}/${expname}/retrain${ver}/
model_module="espnet.nets.pytorch_backend.unsupervised5:E2E"
[ "$beta_ver" == "1" ] && model_module="espnet.nets.pytorch_backend.unsupervised5:E2E"
[ "$beta_ver" == "2" ] && model_module="espnet.nets.pytorch_backend.unsupervised51:E2E"
[ "$beta_ver" == "3" ] && model_module="espnet.nets.pytorch_backend.unsupervised52:E2E"
[ "$beta_ver" == "4" ] && model_module="espnet.nets.pytorch_backend.unsupervised53:E2E"
[ "$beta_ver" == "5" ] && model_module="espnet.nets.pytorch_backend.unsupervised54:E2E"

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
    echo "stage 6: Network Re-Training using CycleGAN inter-domain losses"
    mkdir -p ${expdir}
    ${cuda_cmd} --gpu ${ngpu} ${expdir}/train.log \
        asr_retrain${ver}.py \
        --config ${semi_train_config} \
        --preprocess-conf ${semi_preprocess_config} \
        --init-model ${init_model} \
        --model-module "${model_module}" \
        --train-json ${feat_tr_dir}/data_${bpemode}${nbpe}.json \
        --valid-json ${feat_dt_dir}/data_${bpemode}${nbpe}.json \
        --unsupervised-json ${feat_us_dir}/data_${bpemode}${nbpe}.json \
        --ngpu ${ngpu} \
        --backend ${backend} \
        --outdir ${expdir} \
        --tensorboard-dir tensorboard/${expname} \
        --debugmode ${debugmode} \
        --dict ${dict} \
        --debugdir ${expdir} \
        --minibatches ${N} \
        --verbose -1 \
        --resume ${resume} \
        --seed ${seed} \
        --weight-decay ${weight_decay} \
        --learning-rate ${semi_learning_rate} \
        --unsupervised_loss "${unsupervised_loss}" \
        --supervised-loss-ratio ${sup_loss_ratio} \
        --supervised-data-ratio ${sup_data_ratio} \
        --use-batchnorm ${use_batchnorm} \
        --patience 3 \
        --use_smaller_data_size "False"
    ln -s $expdir exp/${lang}/OurMethod_M1
fi

if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
    echo "stage 7: Decoding and generate label for X'"
    nj=16
    tgtdir=${expdir}
    pids=() # initialize pids
    for rtask in "${unlabel_set}" "${recog_set}" ; do 
    (
        recog_opts=
        if ${skip_lm_training}; then
           lmtag="nolm"
        else
           lmtag="lm"
           recog_opts="--rnnlm ${lmexpdir}/rnnlm.model.best"
        fi

        decode_dir=decode_${rtask}_$(basename ${decode_config%.*})_${lmtag}
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}

        # split data
        splitjson.py --parts ${nj} ${feat_recog_dir}/data_${bpemode}${nbpe}.json

        #### use CPU for decoding
        ngpu=0

        ${decode_cmd} JOB=1:${nj} ${tgtdir}/${decode_dir}/log/decode.JOB.log \
            asr_unsupervised_recog5.py \
            --config ${decode_config} \
            --ngpu ${ngpu} \
            --backend ${backend} \
            --recog-json ${feat_recog_dir}/split${nj}utt/data_${bpemode}${nbpe}.JOB.json \
            --result-label ${tgtdir}/${decode_dir}/data.JOB.json \
            --model ${tgtdir}/${recog_model} \
            ${recog_opts}

       score_sclite.sh --bpe ${nbpe} --bpemodel ${bpemodel}.model --wer true ${tgtdir}/${decode_dir} ${dict}

    )&
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
    echo "Finished"
fi

###### Start NST, so skip_lm_training=false #####
## prepare data and train for M2
skip_lm_training=false
if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ]; then
    echo "stage 8: iterative model training on relablled data"
    for m in {1..8}
    do
        expdir=exp/${lang}/OurMethod_M${m}
        pseudolabel_set=${unlabel_set}_M${m}
        train_set=train_${lang}_init_and_us_M${m}
        decode_dir=decode_${unlabel_set}_decode_lm
        feat_tr_dir=${dumpdir}/${train_set}/delta${do_delta}; mkdir -p ${feat_tr_dir}
        echo "M$((m+1)): Collect pseudo label on M${m} and dump features and make jsons "
        hyp=${expdir}/${decode_dir}/hyp.wrd.trn
        echo $hyp
        [ ! -f $hyp ] && echo "cannot find $hyp" && exit -1
        utils/copy_data_dir.sh data/${unlabel_set} data/${pseudolabel_set}
        cat $hyp | sed 's/\(.*\)(.*/\1/g' > data/${pseudolabel_set}/trans
        cat $hyp | sed 's/.*(\(.*\))/\1/g' | cut -d'-' -f2- > data/${pseudolabel_set}/uttid
        paste -d" " data/${pseudolabel_set}/uttid data/${pseudolabel_set}/trans > data/${pseudolabel_set}/text
        utils/combine_data.sh data/${train_set} data/train_${lang}_init data/${pseudolabel_set}
        fbankdir=fbank
        steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj 40 --write_utt2num_frames true \
            data/${train_set} exp/make_fbank/${train_set} ${fbankdir}
        utils/fix_data_dir.sh data/${train_set}

        dump.sh --cmd "$train_cmd" --nj 4 --do_delta ${do_delta} \
            data/${train_set}/feats.scp data/train_${lang}/cmvn.ark exp/dump_feats/${train_set} \
            ${feat_tr_dir}

        # make json labels
        data2json.sh --feat ${feat_tr_dir}/feats.scp --bpecode ${bpemodel}.model \
            data/${train_set} ${dict} > ${feat_tr_dir}/data_${bpemode}${nbpe}.json

        echo "M$((m+1)): Network Training"
        expdir=exp/${lang}/OurMethod_M$((m+1))
        expname=${lang}_OurMethod_M$((m+1))
        mkdir -p ${expdir}
        ${cuda_cmd} --gpu ${ngpu} ${expdir}/train.log \
            asr_train.py \
            --config ${train_config} \
            --preprocess-conf ${init_preprocess_config} \
            --ngpu ${ngpu} \
            --backend ${backend} \
            --outdir ${expdir}/results \
            --tensorboard-dir tensorboard/${expname} \
            --debugmode ${debugmode} \
            --dict ${dict} \
            --debugdir ${expdir} \
            --minibatches ${N} \
            --verbose 0 \
            --resume ${resume} \
            --learning-rate ${learning_rate} \
            --train-json ${feat_tr_dir}/data_${bpemode}${nbpe}.json \
            --valid-json ${feat_dt_dir}/data_${bpemode}${nbpe}.json

        echo "M$((m+1)): Decoding"
        nj=16
        pids=() # initialize pids
        for rtask in "${unlabel_set}" "${recog_set}" ; do
        (
            recog_opts=
            if ${skip_lm_training}; then
                lmtag="nolm"
            else
                lmtag="lm"
                recog_opts="--rnnlm ${lmexpdir}/rnnlm.model.best"
            fi

            decode_dir=decode_${rtask}_$(basename ${decode_config%.*})_${lmtag}
            feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}

            # split data
            splitjson.py --parts ${nj} ${feat_recog_dir}/data_${bpemode}${nbpe}.json

            #### use CPU for decoding
            ngpu=0

            ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
                asr_recog.py \
                --config ${decode_config} \
                --ngpu ${ngpu} \
                --backend ${backend} \
                --recog-json ${feat_recog_dir}/split${nj}utt/data_${bpemode}${nbpe}.JOB.json \
                --result-label ${expdir}/${decode_dir}/data.JOB.json \
                --model ${expdir}/results/${recog_model} \
                ${recog_opts}

            score_sclite.sh --bpe ${nbpe} --bpemodel ${bpemodel}.model --wer true ${expdir}/${decode_dir} ${dict}

        ) &
        pids+=($!) # store background pids
        done
        i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
        [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
        echo "Finished"
    done
fi


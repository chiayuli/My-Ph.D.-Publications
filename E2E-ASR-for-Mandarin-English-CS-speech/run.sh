#!/usr/bin/env bash

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

# general configuration
backend=pytorch
stage=-1       # start from -1 if you need to start from data download
stop_stage=100
ngpu=1         # number of gpus ("0" uses cpu, otherwise use gpu)
nj=32
debugmode=1
dumpdir=dump   # directory to dump full features
N=0            # number of minibatches to be used (mainly for debugging). "0" uses all minibatches.
verbose=0      # verbose option
resume=        # Resume the training from snapshot

# feature configuration
do_delta=false

#preprocess_config=conf/specaug.yaml
preprocess_config=conf/no_preprocess.yaml
train_config=conf/train_rnn.yaml
lm_config=conf/lm.yaml
decode_config=

# rnnlm related
skip_lm_training=false  # for training & decoding without LM
lm_resume=             # specify a snapshot file to resume LM training
lmtag=                 # tag for managing LMs

# decoding parameter
recog_model=model.acc.best  # set a model to be used for decoding: 'model.acc.best' or 'model.loss.best'
lang_model=rnnlm.model.best # set a language model to be used for decoding

# model average realted (only for transformer)
n_average=5                  # the number of ASR models to be averaged
use_valbest_average=false    # if true, models with top-`n_average` validation/loss are averaged
use_cerbest_average=false    # if true, models with top-`n_average` validation/cer_cer are averaged
                             # if both use_{valbest,cerbest}_average are false, last `n_average` are averaged
# bpemode (unigram or bpe)
nbpe=500
bpemode=unigram

# exp tag
tag="CE_F3_se500" # tag for managing experiments.

. utils/parse_options.sh || exit 1;

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=train_3w_200ENZH # train_clean_100 or train_clean_100_sp
train_dev=dev_3w_200ENZH
recog_set="dev eval"

feat_tr_dir=${dumpdir}/${train_set}/delta${do_delta}; mkdir -p ${feat_tr_dir}
feat_dt_dir=${dumpdir}/${train_dev}/delta${do_delta}; mkdir -p ${feat_dt_dir}
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "stage 1: Feature Generation"
    fbankdir=fbank

    # Generate the fbank features; by default 80-dimensional fbanks with pitch on each frame
    for x in dev_clean test_clean dev_other test_other train_clean_100; do
        steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj ${nj} --write_utt2num_frames true \
            data/${x} exp/make_fbank/${x} ${fbankdir}
        utils/fix_data_dir.sh data/${x}
    done

    # speed perturbation
    mv data/train_clean_100 data/train_clean_100_org
    utils/perturb_data_dir_speed.sh 0.9  data/train_clean_100_org  data/temp1
    utils/perturb_data_dir_speed.sh 1.0  data/train_clean_100_org  data/temp2
    utils/perturb_data_dir_speed.sh 1.1  data/train_clean_100_org  data/temp3
    utils/combine_data.sh --extra-files utt2uniq data/train_clean_100_sp_org data/temp1 data/temp2 data/temp3

    # create dev set
    utils/combine_data.sh --extra_files utt2num_frames data/dev_org data/dev_clean data/dev_other

    # remove utt having more than 3000 frames
    # remove utt having more than 400 characters
    remove_longshortdata.sh --maxframes 3000 --maxchars 400 data/train_clean_100_org data/train_clean_100
    remove_longshortdata.sh --maxframes 3000 --maxchars 400 data/train_clean_100_sp_org data/train_clean_100_sp
    remove_longshortdata.sh --maxframes 3000 --maxchars 400 data/dev_org data/dev
    steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj $nj  --write_utt2num_frames true \
            data/train_clean_100_sp  exp/make_fbank/train_clean_100_sp  ${fbankdir}
    utils/fix_data_dir.sh data/train_clean_100_sp
    # compute global CMVN
    compute-cmvn-stats scp:data/${train_set}/feats.scp data/${train_set}/cmvn.ark

    # dump features for training
    dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta ${do_delta} \
        data/${train_set}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/train ${feat_tr_dir}
    dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta ${do_delta} \
        data/${train_dev}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/dev ${feat_dt_dir}
    for rtask in ${recog_set}; do
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}; mkdir -p ${feat_recog_dir}
        dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta ${do_delta} \
            data/${rtask}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/recog/${rtask} \
            ${feat_recog_dir}
    done
fi


nlsyms=data/lang_char/non_lang_syms_sp_seg2_${tag}.txt
nlsyms_ext=data/lang_char/non_lang_syms_sp_seg2_${tag}_ext.txt
nlsyms_nos=data/lang_char/non_lang_syms_sp_seg2_${tag}_nos.txt
dict=data/lang_char/train_3w_200ENZH_unigram500CE_F3_se500_units.txt
bpemodel=data/lang_char/train_3w_200ENZH_unigram500CE_F3_se500
echo "dictionary: ${dict}"
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    ### Task dependent. You have to check non-linguistic symbols used in the corpus.
    echo "stage 2: Dictionary and Json Data Preparation"
    if [ -z $dict ]; then
        echo "make a non-linguistic symbol list"
        echo "<unk>" > ${nlsyms}
        cut -f 2- data/${train_set}/text | tr " " "\n" | sort | uniq | grep "<" >> ${nlsyms}
        cat ${nlsyms}
        cp ${nlsyms} ${nlsyms_ext}
        echo "▁" >> ${nlsyms_ext}
        echo "▁" > ${nlsyms_nos}

        num_nlsyms=`wc -l ${nlsyms} | cut -d' ' -f1`
        begin_en=$((num_nlsyms+1))
        echo "$num_nlsyms nlsyms"

        # <unk> must be 1, 0 will be used for "blank" in CTC
        echo "make a dictionary"
        echo "train english subword model"
        tmp=data/lang_char/tmp.txt
        cut -f 2- -d" " data/${train_set}/text > $tmp
        filt.py -v -v ${nlsyms} $tmp | env LANG=C grep -oh "[a-zA-Z0-9\']*" > data/lang_char/input.txt
        spm_train --input=data/lang_char/input.txt --vocab_size=${nbpe} --model_type=${bpemode} --model_prefix=${bpemodel} --input_sentence_size=1000000
        spm_encode --model=${bpemodel}.model --output_format=piece < ${inputfile} | tr ' ' '\n' | sort | uniq | grep -v '^▁$' | awk -v num=${num_nlsyms} '{print $0 " " NR+num}' >> ${dict}

        echo "add chinese characters into ${dict}"
        start=`wc -l ${dict} | cut -d' ' -f1`
        env LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8 grep -oP '[\p{Han}]' data/lang_char/input.txt | sort -u | awk '{print $0 " " NR+"'$start'"}' >> ${dict}
        t=`wc -l ${dict} | cut -d' ' -f1`
        echo "▁ $((t+1))" >> ${dict}
    fi

    wc -l ${dict}
    # make json labels
    data2json.sh --nj ${nj} --feat ${feat_tr_dir}/feats.scp --bpecode ${bpemodel}.model \
        data/${train_set} ${dict} > ${feat_tr_dir}/data_${bpemode}${nbpe}.json
    data2json.sh --nj ${nj} --feat ${feat_dt_dir}/feats.scp --bpecode ${bpemodel}.model \
        data/${train_dev} ${dict} > ${feat_dt_dir}/data_${bpemode}${nbpe}.json

    for rtask in ${recog_set}; do
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}
        data2json.sh --nj ${nj} --feat ${feat_recog_dir}/feats.scp --bpecode ${bpemodel}.model \
            data/${rtask} ${dict} > ${feat_recog_dir}/data_${bpemode}${nbpe}.json
    done
fi

if [ -z ${lmtag} ] && ! ${skip_lm_training}; then
    lmtag=$(basename ${lm_config%.*})
fi
# you can skip LM training by setting skip_lm_training=true
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ] && ! ${skip_lm_training}; then
    echo "stage 3: LM Preparation"
    lmexpname=train_rnnlm_${backend}_${lmtag}_${bpemode}${nbpe}_ngpu${ngpu}
    lmexpdir=exp/${lmexpname}
    lmdatadir=data/local/lm_seametrain_aishell_sms_CCG_B2A2B0.7_A2B2A0_ep10_unigram500
    mkdir -p ${lmexpdir}

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
        --dict ${dict} \
        --dump-hdf5-path ${lmdatadir}
fi

if [ -z ${tag} ]; then
    expname=${train_set}_${backend}_$(basename ${train_config%.*})
    if ${do_delta}; then
        expname=${expname}_delta
    fi
    if [ -n "${preprocess_config}" ]; then
        expname=${expname}_$(basename ${preprocess_config%.*})
    fi
else
    expname=${train_set}_${backend}_${tag}
fi
expdir=exp/${expname}
mkdir -p ${expdir}

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "stage 4: Network Training"
    ${cuda_cmd} --gpu ${ngpu} ${expdir}/train.log \
        asr_train.py \
        --config ${train_config} \
        --preprocess-conf ${preprocess_config} \
        --ngpu ${ngpu} \
        --backend ${backend} \
        --outdir ${expdir}/results \
        --tensorboard-dir tensorboard/${expname} \
        --debugmode ${debugmode} \
        --dict ${dict} \
        --debugdir ${expdir} \
        --minibatches ${N} \
        --verbose ${verbose} \
        --resume ${resume} \
        --train-json ${feat_tr_dir}/data_${bpemode}${nbpe}.json \
        --valid-json ${feat_dt_dir}/data_${bpemode}${nbpe}.json
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    echo "stage 5: Decoding"
    if [[ $(get_yaml.py ${train_config} model-module) = *transformer* ]] || \
           [[ $(get_yaml.py ${train_config} model-module) = *conformer* ]] || \
           [[ $(get_yaml.py ${train_config} etype) = custom ]] || \
           [[ $(get_yaml.py ${train_config} dtype) = custom ]]; then
        # Average ASR models
        if ${use_valbest_average}; then
            recog_model=model.val${n_average}.avg.best
            opt="--log ${expdir}/results/log"
        elif ${use_cerbest_average}; then
            recog_model=model.cer${n_average}.avg.best
            opt="--log ${expdir}/results/log --metric cer_ctc"
        else
            recog_model=model.last${n_average}.avg.best
            opt="--log"
        fi
        average_checkpoints.py \
            ${opt} \
            --backend ${backend} \
            --snapshots ${expdir}/results/snapshot.ep.* \
            --out ${expdir}/results/${recog_model} \
            --num ${n_average}

        # LM option
        recog_opts=
        if ${skip_lm_training}; then
            lmtag="nolm"
        else
            recog_opts="--rnnlm ${lmexpdir}/${lang_model}"
        fi
    fi

    pids=() # initialize pids
    for rtask in ${recog_set}; do
    (
        decode_dir=decode_${rtask}_${recog_model}_$(basename ${decode_config%.*})_${lmtag}
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}

        # split data
        splitjson.py --parts ${nj} ${feat_recog_dir}/data_${bpemode}${nbpe}.json

        #### use CPU for decoding
        ngpu=0

        # set batchsize 0 to disable batch decoding
        ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
            asr_recog.py \
            --config ${decode_config} \
            --ngpu ${ngpu} \
            --backend ${backend} \
            --batchsize 0 \
            --recog-json ${feat_recog_dir}/split${nj}utt/data_${bpemode}${nbpe}.JOB.json \
            --result-label ${expdir}/${decode_dir}/data.JOB.json \
            --model ${expdir}/results/${recog_model}  \
            ${recog_opts}


        #score_sclite.sh --bpe ${nbpe} --bpemodel ${bpemodel}.model --wer true ${expdir}/${decode_dir} ${dict}
        score_sclite.sh --nlsyms ${nlsyms_ext} --bpe ${nbpe} --bpemodel ${bpemodel}.model --wer true --ter true ${expdir}/${decode_dir} ${dict}
        score_sclite.sh --nlsyms ${nlsyms_nos} --bpe ${nbpe} --bpemodel ${bpemodel}.model --wer true ${expdir}/${decode_dir}_all ${dict}
    ) &
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
    echo "Finished"
fi

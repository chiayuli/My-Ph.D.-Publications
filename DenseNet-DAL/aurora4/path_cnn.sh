export KALDI_ROOT=`pwd`/../../..
[ -f $KALDI_ROOT/tools/env.sh ] && . $KALDI_ROOT/tools/env.sh
export PATH=$PWD/utils/:$KALDI_ROOT/tools/openfst/bin:$PWD:$PATH
[ ! -f $KALDI_ROOT/tools/config/common_path.sh ] && echo >&2 "The standard file $KALDI_ROOT/tools/config/common_path.sh is not present -> Exit!" && exit 1
. $KALDI_ROOT/tools/config/common_path.sh
export LC_ALL=C
export PYTHONPATH=$PYTHONPATH:`pwd`/pdnnlg_ad/
export PATH=$PYTHONPATH:$PATH
working_dir=exp_pdnn/cnn
gmmdir=exp/tri2b_multi
gpu=gpu
pythonCMD=python
num_pdfs=`gmm-info $gmmdir/final.mdl | grep pdfs | awk '{print $NF}'`
feat_dim=$(gunzip -c $working_dir/train_multi_tr95.pfile.1.gz |head |grep num_features| awk '{print $2}')

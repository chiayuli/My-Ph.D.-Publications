. ./path_cnn_dn.sh

echo =====================================================================
echo "                Dump Convolution-Layer Activation                  "
echo =====================================================================
mkdir -p $working_dir/data_conv_d
#for set in test_eval92_A test_eval92_B test_eval92_C test_eval92_D test_eval92 ; do
for set in test_eval92 ; do
  if [ ! -d $working_dir/data_conv_d/$set ]; then
    steps_pdnnlg/make_conv_feat_densenet_L61_12_test2_theta0.4.sh --nj 5 $working_dir/data_conv_d/$set $working_dir/data/$set $working_dir $working_dir/nnet.param.dn \
      $working_dir/nnet.cfg.dn $working_dir/_log_d $working_dir/_conv_d || exit 1;
    # Generate *fake* CMVN states here.
    steps/compute_cmvn_stats.sh --fake \
      $working_dir/data_conv_d/$set $working_dir/_log_d $working_dir/_conv_d || exit 1;
  fi  

echo =====================================================================
echo "                           Decoding                                "
echo =====================================================================
# In decoding, we take the convolution-layer activation as inputs and the 
# fully-connected layers as the DNN model. So we set --norm-vars, --add-deltas
# and --splice-opts accordingly.
if [ ! -f  $working_dir/decode.done ]; then
  cp $gmmdir/final.mdl $working_dir || exit 1;  # copy final.mdl for scoring
  graph_dir=$gmmdir/graph
  steps_pdnnlg/decode_densenet_0.1.sh --nj 5 --scoring-opts "--min-lmwt 1 --max-lmwt 18" --norm-vars false --add-deltas false --splice-opts "--left-context=0 --right-context=0" $graph_dir $working_dir/data_conv_d/$set ${gmmdir}_ali_tr95 $working_dir/decode_bd_tgpr_${set}_densenet_L61_12_test2_theta0.4_acwt0.1 || exit 1;
fi
done

echo "Finish !!"

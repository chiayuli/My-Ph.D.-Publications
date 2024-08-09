#!/bin/bash
. path.sh
. cmd.sh


lang=hun
unpair_set=train_${lang}_us
dir=hun_mixed_2012_10K # the directory store external text from https://wortschatz.uni-leipzig.de/en/download/German#deu_news_2023 
                       # e.g. hun is hungarian
audio_root=$(pwd)/downloads 
tgtdir=data/train_$lang

. utils/parse_options.sh || exit 1;

mkdir -p $tgtdir/.tmp

# prepare kaldi format text by defining utterance ID, normalizing text, and remove non-alphabetic characters
CHARS=$(printf "%b" "\U00A0\U1680\U180E\U2000\U2001\U2002\U2003\U2004\U2005\U2006\U2007\U2008\U2009\U200A\U200B\U202F\U205F\U3000\UFEFF")
cut -f2- $dir/hun_mixed_2012_10K-sentences.txt | grep -v -P "[0-9]+" | tr "[:lower:]" "[:upper:]" > $tgtdir/.tmp/train_mixed.txt
awk -v id=0 '{printf("%.5d-LeipzigCorpus", id);id=id+1;for(i=1;i<=NF;i++){printf(" %s",$i)}printf("\n")}' $tgtdir/.tmp/train_mixed.txt > $tgtdir/text
sed -i 's/['"$CHARS"']/ /g' $tgtdir/text

# prepare uttt2spk
awk -v id=0 '{printf("%.5d-LeipzigCorpus %.5d\n", id,id);id=id+1}' $tgtdir/.tmp/train_mixed.txt > $tgtdir/utt2spk

# prepare fake wav.scp contains unpaired audio
total=`wc -l text | cut -d" " -f1`
cut -d" " -f4 data/${unpair_set}/wav.scp > $tgtdir/.tmp/${unpair_set}.audio
cat $tgtdir/.tmp/${unpair_set}.audio $tgtdir/.tmp/${unpair_set}.audio | shuf -n ${total} > $tgtdir/.tmp/${total}.audio
mkdir fake_audio && cd fake_audio
c=1;
for uttid in `cut -d" " -f1 $tgtdir/text`;
do
  wav=`head -n $c $tgtdir/.tmp/${total}.audio | tail -n 1`
  c=$((c+1)); echo $uttid $audio_root/$wav >> $tgtdir/.tmp/utt2wav
  ln -s $audio_root/$wav ${uttid}.mp3
done

awk -v audio_root="${audio_root}" '{printf("%s ffmpeg -i %s/%s.mp3 -f wav -ar 16000 -ab 16 -ac 1 - |\n",$1,audio_root,$1)}' $tgtdir/text > $tgtdir/wav.scp

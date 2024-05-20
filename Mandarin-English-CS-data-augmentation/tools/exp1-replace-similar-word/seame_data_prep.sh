#!/bin/bash
#. path.sh
#. cmd.sh
export PATH=$PATH:$E2E/tools/stanford-segmenter-2018-10-16/
srcdir=/resources/corpora/LDC2015S04/seame/
dir=data/local/all
tmpdir=data/tmp
train=data/train
dev=data/dev
eval=data/eval
mkdir -p $train
mkdir -p $dev
mkdir -p $eval
mkdir -p $tmpdir
mkdir -p $dir
stage=1
if [ $stage == 1 ];then
	## PhaseII DATA
	awk '{ 
		name=$1; stime=$2; etime=$3; type=$4;
	   	if ( NF > 4 )
	   	{
			printf("%s_%08.0f-%08.0f %s", 
              name, int(stime), int(etime), type);
       		for(i=5;i<=NF;i++) printf(" %s", $i); printf "\n"
		}
	}' $srcdir/data/*/transcript/phaseII/*.txt  > $dir/transcripts_p2.txt

	## remove some weird transcriptions
	cat $dir/transcripts_p2.txt | grep -vE '13NC26MBQ_0101_02793853-02795426' > $dir/transcripts_p2_v1.txt
fi
## remove fullwidth punctuations
#LANG=C sed -E 's/[\uff01-\uff5e]/\=nr2char(char2nr(submatch(0))-65248)/g' < $dir/transcripts_p2_v1.txt > $dir/test
stage=2
if [ $stage == 2 ]; then
	if [ 1 == 1 ]; then
	# remove punctuations
	cat $dir/transcripts_p2_v1.txt | cut -d' ' -f1  > $dir/utts
	cat $dir/transcripts_p2_v1.txt | cut -d' ' -f2  > $dir/types
	cat $dir/transcripts_p2_v1.txt | cut -d' ' -f3- > $dir/trans
	sed -i 's/-/ /g' $dir/trans
	sed -i 's/=/ /g' $dir/trans
	sed -i 's/×/ /g' $dir/trans
	sed -i 's/——/ /g' $dir/trans
	sed -i 's/\[hu\[h/\[huh\]/g' $dir/trans
	sed -i 's/\[\[/\[/g' $dir/trans
	sed -i 's/\]\]/\]/g' $dir/trans
	sed -i 's/\[[^]]+\[/<dis-par>/g' $dir/trans
	sed -i 's/\][^[]+\]/<dis-par>/g' $dir/trans
	sed -i 's/(\[/(/g' $dir/trans
	sed -i 's/ ( / /g' $dir/trans
	sed -i 's/\])/)/g' $dir/trans
	sed -i 's/}/]/g' $dir/trans
	sed -i 's/{/[/g' $dir/trans
	sed -i 's/】/]/g' $dir/trans
	sed -i 's/【/[/g' $dir/trans
	sed -iE 's/[·?、%#`,."~@*]//g' $dir/trans
	paste -d' ' $dir/utts $dir/types $dir/trans > $dir/transcripts_p2_v1.txt
	rm $dir/{utts,types,trans,transE}

	# replace non-speech (xxx) by <non-speech>
	cat $dir/transcripts_p2_v1.txt | sed -E 's/[(][a-zA-Z0-9 ]+[)]/ <non-speech> /g' > $dir/transcripts_p2_v2.txt
	LANG=C sed -E 's/[(](\xe4[\xb8-\xbf][\x80-\xbf]|[\xe5-\xe9][\x80-\xbf][\x80-\xbf]|[a-z ])+[)]/ <non-speech> /g' < $dir/transcripts_p2_v2.txt > $dir/transcripts_p2_v2_1.txt
	# some imcomplete non-speech labels need to be replaced by <non-speech>
	sed -i 's/(ppl0/ <non-speech> /g' $dir/transcripts_p2_v2_1.txt
	sed -i 's/<ppo>/ <non-speech> /g' $dir/transcripts_p2_v2_1.txt
	sed -i 's/(ppl/ <non-speech> /g' $dir/transcripts_p2_v2_1.txt
	sed -i 's/(ppo/ <non-speech> /g' $dir/transcripts_p2_v2_1.txt
	sed -i 's/(erm/ <non-speech> /g' $dir/transcripts_p2_v2_1.txt
	sed -i 's/ppb)/ <non-speech> /g' $dir/transcripts_p2_v2_1.txt
	#sed -i 's/()//g' $dir/transcripts_p2_v2_1.txt
	sed -i 's/(//g' $dir/transcripts_p2_v2_1.txt
	sed -i 's/)//g' $dir/transcripts_p2_v2_1.txt

	# replace chinese discourse particle by <dis-par>
	LANG=C sed -E 's/\[(\xe4[\xb8-\xbf][\x80-\xbf]|[\xe5-\xe9][\x80-\xbf][\x80-\xbf])+\]/ <dis-par> /g' < $dir/transcripts_p2_v2_1.txt > $dir/transcripts_p2_v2_2.txt
	#LANG=C sed -E 's/\[(\xe4[\xb8-\xbf][\x80-\xbf]|[\xe5-\xe9][\x80-\xbf][\x80-\xbf])/ <dis-par> /g' < $dir/transcripts_p2_v2_2.txt > $dir/transcripts_p2_v2_3.txt
	# singapore discourse particle, like "[lah]" or "[loh]", need to be replaced by <dis-par>
	# some imcomplete discourse particle, like "[lah" or "loh]", need to be replaced by <dis-par>
    cat $dir/transcripts_p2_v2_2.txt | sed -E 's/\[[^ []+\]/ <dis-par> /g' | sed -E 's/\[[^ []+ / <dis-par> /g' | sed -E 's/\[[^ []+$/ <dis-par> /g' | sed -E 's/ [^ []+\]/ <dis-par> /g' | sed -E 's/\[[^ ]]+\[/ <dis-par> /g' > $dir/transcripts_p2_v2_4.txt
	sed -i 's/\[//g' $dir/transcripts_p2_v2_4.txt
	sed -i 's/\]//g' $dir/transcripts_p2_v2_4.txt

    # unify the <unk> symbol
	sed -i 's/<unk>/ <aunk> /g' $dir/transcripts_p2_v2_4.txt
	sed -i 's/< unk >/ <aunk> /g' $dir/transcripts_p2_v2_4.txt
	sed -i 's/<unk >/ <aunk> /g' $dir/transcripts_p2_v2_4.txt

	# remove some bad transctiptions (it contains <unk movie title>, <unk place>, <unk job> )
    grep -vE "01NC02FBY_0101_06775608-06777105|03NC05FAX_0201_02590513-02594063|03NC05FAX_0201_03782411-03783680|03NC05FAX_0201_03784315-03786422|33NC43FBQ_0101_01797447-01798486|NI34FBQ_0101_00704348-00712475" $dir/transcripts_p2_v2_4.txt > $dir/transcripts_p2_v2_5.txt
    fi

	# segmentation
	cut -d' ' -f1 $dir/transcripts_p2_v2_5.txt > $dir/utts
	cut -d' ' -f2 $dir/transcripts_p2_v2_5.txt > $dir/types
	cut -d' ' -f3- $dir/transcripts_p2_v2_5.txt | sed 's/^ //g' | tr -s ' ' > $dir/trans
	# insert space between english and mandarin words
	python3 ../../../src/utils/splitenzh.py -f $dir/trans > $dir/trans.n
	# insert space between each mandarin words
	perl -CIOED -p -e 's/\p{Block=CJK_Unified_Ideographs}/ $& /g' $dir/trans.n > $dir/trans
	# number to words
	digits=( "" one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eightteen nineteen )
	for d in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 ;
	do
		sed -i "s/^$d /${digits[d]} /g" $dir/trans
		sed -i "s/ $d / ${digits[d]} /g" $dir/trans
		sed -i "s/ ${d}$/ ${digits[d]}/g"  $dir/trans
	done
	sed -i "s/35/thirty five/g" $dir/trans
	sed -i "s/50/fifty/g" $dir/trans
	#sed -i "s/wall 这样 0/wall 这样/g" $dir/trans
	# replace "i' m" by "i'm"
	# remove utterance without empty transcription
	paste -d' ' $dir/utts $dir/types $dir/trans | sed "s/' /'/g" | awk '{if( NF > 1){printf("%s", $1);for(i=2;i<=NF;i++)printf(" %s", $i);printf("\n")}}' > $dir/transcripts_p2_final.txt
	paste -d' ' $dir/utts $dir/trans | sed "s/' /'/g" | awk '{if( NF > 1){printf("%s", $1);for(i=2;i<=NF;i++)printf(" %s", $i);printf("\n")}}' > $dir/text_final.txt
	rm $dir/{utts,trans,trans.n}
	mkdir -p $dir/tmp
	#mv $dir/transcripts_p2_v2_*.txt $dir/tmp/
fi

# check whether utterance exists or not
[ -f $dir/pattern ] && rm $dir/pattern
touch $dir/pattern
for f in `ls $srcdir/data/*/transcript/*/*.txt`;
do
	recordid=`echo $(basename $f) | sed 's/.txt//g'`
	found=`find $srcdir/data/*/audio -name $recordid.flac`
	if [ -z "$found" ]; then
		echo "$recordid doesn't exist!!"
		echo $recordid >> $dir/pattern
	fi
done

grep -vf $dir/pattern $dir/text_final.txt > $dir/text

# Make segment files from transcript
# segments file format is: utt-id recording-id start-time end-time, e.g.:
# 01NC01FBX_0101_00086300-00088370 01NC01FBX_0101 86.3 88.37
awk '{  
       segment=$1;
       split(segment,S,"[_-]");
       audioname=S[1]; part=S[2]; startf=S[3]; endf=S[4];
       print segment " " audioname "_" part " " startf/1000 " " endf/1000
}' < $dir/text > $dir/segments


# Make wav.scp file
# wav.scp format is: recording-id extended-filename
# 46NC41MBP_0101 ffmpeg -loglevel -8 -i /mount/projekte/slu/Data/SEAME/data/conversation/audio/46NC41MBP_0101.flac -f wav - |

[ -f $dir/wav.scp ] && rm $dir/wav.scp
for f in `ls $srcdir/data/*/audio/*.flac`; 
do 
	name=`echo $(basename $f) | sed 's/.flac//g'` 
	found=`grep "$name" $dir/segments`
	if [[ $found = "" ]]
	then
		echo "$name audio exist but no transcription."
	else
		echo $name" ffmpeg -loglevel -8 -i $f -f wav - |" >> $dir/wav.scp
	fi
done 

cp $dir/{wav.scp,segments,text} $tmpdir

# Make utt2spk and spk2utt
cut -d' ' -f1 $tmpdir/text > $tmpdir/utts
cat $tmpdir/utts | cut -d'_' -f1 > $tmpdir/spkids
paste -d' ' $tmpdir/utts $tmpdir/spkids > $tmpdir/utt2spk
utils/utt2spk_to_spk2utt.pl $tmpdir/utt2spk > $tmpdir/spk2utt

## split data into train, dev and eval by spkLst.train spkLst.dev spkLst.eval
for set in train dev eval; 
do
	[ -d data/${set} ] && rm -rf data/${set}
	mkdir -p data/${set}
	grep -f spkLst.${set} $tmpdir/wav.scp > data/${set}/wav.scp
	grep -f spkLst.${set} $tmpdir/utt2spk > data/${set}/utt2spk
	grep -f spkLst.${set} $tmpdir/text > data/${set}/text
	grep -f spkLst.${set} $tmpdir/segments > data/${set}/segments
	utils/utt2spk_to_spk2utt.pl data/${set}/utt2spk > data/${set}/spk2utt
	utils/validate_data_dir.sh --no-feats data/${set}
done

## in dev and eval, maybe you don't want to evaluate the utterance which only contains non-speech.
for set in dev eval;
do
	grep -E "[0-9]( <dis-par>)*( <aunk>)*( <non-speech>)*$" data/$set/text | cut -d' ' -f1 > ${set}_nonsp.list
	mkdir -p data/${set}_r
	for f in wav.scp utt2spk segments text ;
	do
		grep -vf ${set}_nonsp.list data/${set}/$f > data/${set}_r/$f
	done
	utils/utt2spk_to_spk2utt.pl data/${set}_r/utt2spk > data/${set}_r/spk2utt
	utils/validate_data_dir.sh --no-feats data/${set}_r
done

echo "Data preparation for PhaseII is Finished"

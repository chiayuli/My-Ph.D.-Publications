less -c ../tr05_BUS_SIMU/utt2spk | cut -d' ' -f1 | cut -d'_' -f2 | tr '[:upper:]' '[:lower:]' > bus_simu_spk_list

for f in "wav.scp" "utt2spk" "text"; 
do 
  grep -f bus_simu_spk_list $f > tmp;
  mv tmp $f; 
done

## count the max number of frames per utterance
for set in "trainA" "trainB" "valA" "valB" "testA" "testB"
do
  less -c $set/utt2num_frames | awk -v max=0 '{if($2>max){max=$2;print $1,max}}END{print max}'
done

## prepare aligned data
cut -d' ' -f1 trainB_v7/wav.scp > utt.list
while read line;
do
    echo $line
    new=`grep -i "$line" wav.scp | head -n 1 | cut -d' ' -f2-` ; echo "$line $new" >> wav.scp.n5
done < utt.list
mv wav.scp.n5 wav.scp

## prepare aligned data based on trainA
cd trainA_v9
while read line ; 
do 
    echo $line; 
    keyword=`echo $line | cut -d' ' -f1 | cut -d'_' -f2 | tr '[:upper:]' '[:lower:]'`; 
    wav=`grep $keyword ../trainB_v9/wav.scp | cut -d' ' -f2-`; 
    uttid=`echo $line | cut -d' ' -f1`; 
    tmp=`echo $wav | sed 's/|/> /g'`; 
    cmd=`echo "$tmp /mount/arbeitsdaten/asr-3/licu/kaldi-cuda/egs/chime3/s5_fb40/clean_v9/${uttid}_CLEAN.wav"`; 
    eval $cmd;
    echo "${uttid}_CLEAN /mount/arbeitsdaten/asr-3/licu/kaldi-cuda/egs/chime3/s5_fb40/clean_v9/${uttid}_CLEAN.wav" >> wav.scp.forB; 
done < wav.scp

less -c utt2spk | sed 's/ /_CLEAN /g' > ../trainB_v9/utt2spk
less -c text | sed 's/_SIMU/_SIMU_CLEAN/g' | sed 's/_REAL/_REAL_CLEAN/g' > ../trainB_v9/text

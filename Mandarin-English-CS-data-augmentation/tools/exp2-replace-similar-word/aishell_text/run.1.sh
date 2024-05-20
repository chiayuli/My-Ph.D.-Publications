#!/bin/bash
source /mount/arbeitsdaten/asr-3/licu/download/venv/bin/activate
export GOOGLE_APPLICATION_CREDENTIALS="/mount/arbeitsdaten/asr-3/licu/download/EXP2/data/My_First_Project-d1f7fca67c76.json"
cmd=../replace_similar_word_v12.py

dir=$1
name="aishell1"
count=1
mkdir -p $dir/exp
mkdir -p $dir/log

for f in `find $dir -name "x*"`;
do
  output=${name}_$(basename $f)
  echo "make cs text from $f and save to $dir/exp/$output"
  nohup python -u $cmd -t $f -o $dir/exp/$output > $dir/log/run.log.${output}
done

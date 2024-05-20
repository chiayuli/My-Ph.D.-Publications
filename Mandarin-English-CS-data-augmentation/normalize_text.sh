text=$1
expdir=$2

if [ ${#} -ne 2 ]; then
  echo "./normalize_text.sh <text> <expdir>"
  echo "./normalize_text.sh exp/9_CNN_valid_text exp/normalized_text/v7"
  exit 1
fi

echo "input text is ${text}"
echo "output text is ${expdir}/$(basename $text)_norm1"

less -c $text | sed '/^$/d' | sed 's/<eos>//g' | sed 's/<sos>//g' | sed "s/&#39;/\'/g" | tr "[:upper:]" "[:lower:]" | sed 's/[?.,!]//g' | sed 's/^ //g' | sed '/^$/d' > ${expdir}/$(basename $text)_norm1
# insert space between mandarin character
perl -CIOED -p -e 's/\p{Block=CJK_Unified_Ideographs}/ $& /g' ${expdir}/$(basename $text)_norm1 | tr -s ' ' | sed 's/^ //g' > ${expdir}/$(basename $text)_norm2

text=$1

less -c $text | sed '/^$/d' | sed "s/&#39;/\'/g" | tr "[:upper:]" "[:lower:]" | sed 's/[?.,!]//g' | sed 's/^ //g' | sed '/^$/d' > norm_${text}
# insert space between mandarin character
perl -CIOED -p -e 's/\p{Block=CJK_Unified_Ideographs}/ $& /g' norm_${text} | tr -s ' ' | sed 's/^ //g' > Final_norm_${text}

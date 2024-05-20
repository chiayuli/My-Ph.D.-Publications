#!/bin/sh
muse_en_dic=/mount/arbeitsdaten/asr-3/licu/download/MUSE/vector-en.tsv
muse_zh_dic=/mount/arbeitsdaten/asr-3/licu/download/MUSE/vector-zh.tsv
data=data
#rm -rf $data
#mkdir $data

tvsub=/mount/arbeitsdaten/asr-3/licu/download/tvsub
#word2vec=/mount/arbeitsdaten44/projekte/asr-3/licu/download/MUSE/dumped/debug/3owial0yuq # supervised src: zh, tgt: en
word2vec=/mount/arbeitsdaten44/projekte/asr-3/licu/download/MUSE/dumped/debug/5qv4k2xkyk # supervised src: en, tgt: zh
#word2vec=/mount/arbeitsdaten44/projekte/asr-3/licu/download/MUSE/dumped/debug/1nbphj9n1r # unsupervised 
#word2vec=/mount/arbeitsdaten44/projekte/asr-3/licu/download/MUSE/dumped/debug/4qsco8if7z # unsupervised 
w2v_dim=300
biexp=data/muse/sup
f="tvsub-vectors-sup-en-zh.txt"

# train w2v by MUSE
    #cut -d' ' -f2- ../../data/tvsub_en_text_num_2ndNorm | tr " " "\n" | sort -u | awk '{printf("%s \n",$1)}' > tvsub_en_list

    #sed 's/^/ /g' wiki.en.vec > wiki.en.vec.tmp
    #cut -d' ' -f2- ../../data/tvsub_en_text_num_2ndNorm | tr " " "\n" | sort -u | awk '{printf(" %s \n",$1)}' > tvsub_en_list
    #grep -f tvsub_en_list wiki.en.vec.tmp > tvsub_vectors/tvsub.en.vec
    #n=`wc -l tvsub_vectors/tvsub.en.vec | cut -d' ' -f1`
    #echo "$n $wav_dim" | cat - tvsub_vectors/tvsub.en.vec > tvsub_vectors/tmp
    #mv tvsub_vectors/tmp tvsub_vectors/tvsub.en.vec
    #sed -i 's/^ //g' tvsub_vectors/tvsub.en.vec

    #grep -f tvsub_zh_list wiki.zh.vec > tvsub_vectors/tvsub.zh.vec
    #n=`wc -l tvsub_vectors/tvsub.zh.vec | cut -d' ' -f1`
    #echo "$n $wav_dim" | cat - tvsub_vectors/tvsub.zh.vec > tvsub_vectors/tmp
    #mv tvsub_vectors/tmp tvsub_vectors/tvsub.zh.vec

# generate chinese+english dictionary
    # combine vector-en.tsv and vector-zh.tsv
    #cut -f1 ${muse_zh_dic} | grep -P "[\p{Han}]" | grep -vP "[\p{Hiragana}]" | grep -vP "[\p{Katakana}]" | sort -u > $data/zh_list
    #cut -f1 ${muse_en_dic} | iconv -f UTF-8 -t ASCII//TRANSLIT | grep -E "^[a-z]+$" | grep -v "[[:punct:]]" | sort -u > $data/en_list
    #grep -f $data/zh_list ${muse_zh_dic} > $data/zh-vector
    #grep -f $data/en_list ${muse_en_dic} > $data/en-vector
    #cat $data/zh-vector $data/en-vector > $data/zhen-vector
    #echo "<unk> 0" > $data/zhen_dictionary
    #cat $data/zh_list $data/en_list | awk -v i=1 '{printf("%s %s\n",$1,i);i=i+1}' >> $data/zhen_dictionary
    cat $data/zh_list $data/en_list | awk '{printf("%s \n",$1)}' > $data/zhen_list
    #grep -f $data/zh_list $word2vec/vectors-zh.txt > $biexp/$f
    #grep -f $data/en_list $word2vec/vectors-en.txt >> $biexp/$f
    #n=`wc -l $biexp/$f | cut -d' ' -f1`
    #echo "$n $w2v_dim" | cat - $biexp/$f > $biexp/tmp
    #mv $biexp/tmp $biexp/$f

    # check OOV in tvsub 
    # 1. generate english words list from tvsub
    #nl -nrz -w8 ${tvsub}/data/processed/train/train.en | sed 's/\t/ /g' > $data/tvsub_en_text_num
    #less -c $data/tvsub_en_text_num | sed 's/&apos;ve/have/g' | sed 's/&apos;ll/will/g' | sed 's/&apos;re/are/g' | sed 's/&apos;s/is/g' | sed "s/&apos;t/\'t/g" | sed "s/n\'t/not/g" | sed "s/&apos;m/am/g" | sed 's/&apos;d/would/g' | sed 's/&quot;//g' |sed 's/&apos;//g' | tr -d "[:punct:]" > $data/tvsub_en_text_num_firstNorm

    # 2. generate chinese words list from tvsub
    #nl -nrz -w8 ${tvsub}/data/processed/train/train.zh | sed 's/\t/ /g' | sed 's/&quot;//g' | sed 's/&amp;//g' | tr -d '[:punct:]' > $data/tvsub_zh_text_num_firstNorm

    # 3. generate chinese and english words list from tvsub
    #python3 preprocess_paired_data.py -f1 $data/tvsub_zh_text_num_firstNorm -f2 $data/tvsub_en_text_num_firstNorm > $data/log&
    #mv english_norm.txt chinese_norm.txt $data/

    #cut -d' ' -f1 $data/chinese_norm.txt | sort -u > $data/chi_id
    #cut -d' ' -f1 $data/english_norm.txt | sort -u > $data/eng_id
    #comm -12 $data/chi_id $data/eng_id > $data/keep_id
    #grep -f $data/keep_id $data/chinese_norm.txt > $data/tvsub_zh_text_num_2ndNorm
    #grep -f $data/keep_id $data/english_norm.txt > $data/tvsub_en_text_num_2ndNorm

    # 3. check OOV in tvsub
    #cut -d' ' -f2- $data/tvsub_zh_text_num_2ndNorm | tr " " "\n" | sort -u | awk '{printf("%s \n",$1)}' > $data/tvsub_all_words
    #cut -d' ' -f2- $data/tvsub_en_text_num_2ndNorm | tr " " "\n" | sort -u | awk '{printf("%s \n",$1)}' >> $data/tvsub_all_words
    #grep -vf $data/zhen_list $data/tvsub_all_words | sort -u | sed 's/\(.*\)/ \1/g' > $data/tvsub_oov
    #n=`wc -l $data/tvsub_oov | cut -f1`
    #echo "there are $n oov in tvsub/train"
    #grep -f $data/tvsub_oov $data/tvsub_zh_text_num_2ndNorm | cut -d' ' -f1 > $data/oov_remove.list
    #grep -f $data/tvsub_oov $data/tvsub_en_text_num_2ndNorm | cut -d' ' -f1 >> $data/oov_remove.list
    #grep -vf $data/oov_remove.list $data/tvsub_zh_text_num_2ndNorm > $data/tvsub_zh_text_num_3rdNorm
    #grep -vf $data/oov_remove.list $data/tvsub_en_text_num_2ndNorm > $data/tvsub_en_text_num_3rdNorm
    
    # check OOV in SEAME
    # 1. generate english words list from SEAME
    # 2. generate chinese words list from SEAME
    # 3. check OOV in SEAME
    less -c $E2E/egs/cs/asr5_0/data/local/all/transcripts_p2_v2_5.txt | grep " CS " | cut -d' ' -f3- | sed 's/<dis-par>//g' | sed 's/<non-speech>//g' | sed 's/<aunk>//g' | sed 's/<unk>//g' | sed 's/<unl>//g' | tr -s " " | sed 's/^ //g' > SEAME_text
    python3 $E2E/src/utils/splitenzh.py -f SEAME_text > SEAME_text_1
    less -c SEAME_text_1 | tr " " "\n" | sort -u | sed '/^$/d' | awk '{printf("%s \n",$1)}' > SEAME_words_list
    grep -vf $data/zhen_list SEAME_words_list | sed 's/\(.*\)/ \1/g' > SEAME_oov
    grep -vf SEAME_oov SEAME_text_1 | sed  '/^$/d' > SEAME_text_2
    
 #model = gensim.models.KeyedVectors.load_word2vec_format("MUSE/data/wiki.en.vec")
#save_word2vec_format


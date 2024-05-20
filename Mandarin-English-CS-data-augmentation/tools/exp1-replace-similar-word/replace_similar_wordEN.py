# -*- coding: utf-8 -*-
import argparse
import re
import gensim
import string
import random

def parse():
    ap=argparse.ArgumentParser()
    ap.add_argument('-t', '--text', required=True, help='the CS text')
    ap.add_argument('-l', '--lang', default='zh', help='english (en) or chinese (zh)')    
    ap.add_argument('-o', '--output', required=True, help='the output file of augmented CS text')    
    args=ap.parse_args()
    return args

def noPunc(words):
    # return true if there is no punctuation in the string
    for w in words:
      if w in string.punctuation:
        return False
      elif "—" in w:
        return False
    return True

def augment(line, model_en, model_man):
    # split string by word
    words = re.findall(r'[a-z]+|[\u4e00-\ufaff]+', line, re.UNICODE)
    strings=[]
    toplist_en=[]
    toplist_man=[]
    langs=[]
    index_en=-1
    index_man=-1
  
    for i in range(len(words)):
      if re.match(r'[a-z]+',words[i]): # 0 denotes English word
        langs.append(0)   
      elif re.match(r'[\u4e00-\ufaff]+',words[i]): # 1 denotes Mandarin word
        langs.append(1)
   
    if langs.count(0) == 1: # replace English word
      if model_en is None:
        logging.error('English word2vec model is None')
        return -1
      else:
        if words[langs.index(0)] in model_en.vocab:
          index_en = langs.index(0)
          similar_words=model_en.similar_by_word(word=words[index_en], topn=30)
          print(similar_words)
          # make sure only English words are selected
          for u in similar_words:
            s = str(u[0])
            if re.match(r'[a-z]+', s) and noPunc(s):
              print('match '+str(u[0]))
              toplist_en.append(str(u[0]))
            else:
              print('not match '+str(u[0]))
          print(toplist_en)

    
    strings.append(line)
    for w in toplist_en:
      if index_en != (len(words) - 1):
        string = ' '.join(words[:index_en]) + ' ' + w + ' ' + ' '.join(words[index_en+1:]) + '\n'
      else:
        string = ' '.join(words[:index_en]) + ' ' + w + '\n'
      strings.append(string)
   
    return strings

if __name__ == '__main__':
    args=parse()
    model_man = None
    model_en = None

    #model_man=gensim.models.KeyedVectors.load_word2vec_format('wiki.zh.vec')
    model_en=gensim.models.KeyedVectors.load_word2vec_format('wiki.en.vec')
    
    with open(args.text, 'r') as f:
        with open(args.output, 'w') as out:
            for line in f:
                strings=augment(line, model_en, model_man)
                i=0
                for s in strings:
                  if i == 3:
                    break 
                  out.write(s)
                  i=i+1

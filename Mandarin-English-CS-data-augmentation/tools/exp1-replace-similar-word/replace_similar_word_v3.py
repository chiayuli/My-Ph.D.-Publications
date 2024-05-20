# -*- coding: utf-8 -*-
import datetime
import argparse
import re
import gensim
import string
import random

def parse():
    ap=argparse.ArgumentParser()
    ap.add_argument('-t', '--text', required=True, help='the CS text')
    #ap.add_argument('-l', '--lang', default='zh', help='english (en) or chinese (zh)')    
    ap.add_argument('-o', '--output', required=True, help='the output file of augmented CS text')    
    ap.add_argument('-l', '--length', default=2, help='how many words in a sentence')
    ap.add_argument('-n', '--number', default=2, help='how many synthetic sentences per line')    
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

def augment(line, model_en, model_man, length):
    # split string by word
    n=10
    words = re.findall(r'[a-z]+|[\u4e00-\ufaff]+', line, re.UNICODE)
    strings=[]
    string=''
    toplist_en=[]
    toplist_man=[]
    langs=[]
    index_en=-1
    index_man=-1

    strings.append(line)
    print('len '+str(len(words)))
    if len(words) > int(length):
        return strings
    for i in range(len(words)):
      if re.match(r'[a-z]+',words[i]): # 0 denotes English word
        langs.append(0)   
      elif re.match(r'[\u4e00-\ufaff]+',words[i]): # 1 denotes Mandarin word
        langs.append(1)
   
    # replace random English word by its most_siimilar word
    if langs.count(0) >= 1 : # replace English word
      if model_en is None:
        logging.error('English word2vec model is None')
        return -1
      else:
        index_en = langs.index(0)
        if words[index_en] in model_en.vocab:
          similar_words=model_en.similar_by_word(word=words[index_en], topn=n)
          #print(words[index_en])
          #print(similar_words)
          # make sure only English words are selected
          for u in similar_words:
            s = str(u[0])
            if re.match(r'[a-z]+', s) and noPunc(s):
              toplist_en.append(str(u[0]))
    
    # replace random Mandarin word by its most_siimilar word
    if langs.count(0) > 0 and langs.count(1) > 0: # replace Mandarin word
      random.seed(100)
      i=0
      while i <= len(langs):
        i+=1
        index_man = random.randrange(len(langs))
        if ( langs[index_man] == 1 ) and ( words[index_man] in model_man.vocab ) : # make sure it is Mandarin word and not OOV for model
          similar_words=model_man.similar_by_word(word=words[index_man], topn=n)
          #print(words[index_man])
          #print(similar_words)
          # make sure only Mandarin words are selected
          for u in similar_words:
            s = str(u[0])
            if re.match(r'[\u4e00-\ufaff]+', s) and noPunc(s):
              #print('match '+str(u[0]))
              toplist_man.append(str(u[0]))
            #else:
              #print('not match '+str(u[0]))
          #print(toplist_man)
          break

    # there's new alternative English word
    if len(toplist_en) > 0 :
      if index_en != (len(words) - 1):
        string = ' '.join(words[:index_en]) + ' ' + str(toplist_en[0]) + ' ' + ' '.join(words[index_en+1:]) + '\n'
      else:
        string = ' '.join(words[:index_en]) + ' ' + str(toplist_en[0]) + '\n'
    
    # there's alternative Mandarin word
    if string is not '':
      words = re.findall(r'[a-z]+|[\u4e00-\ufaff]+', string, re.UNICODE)
    if len(toplist_man) > 0 :
      if index_man != (len(words) -1) :
        string = ' '.join(words[:index_man]) + ' ' + str(toplist_man[0]) + ' ' + ' '.join(words[index_man+1:]) + '\n'
      else:
        string = ' '.join(words[:index_man]) + ' ' + str(toplist_man[0]) + '\n'
    if string != '':
      strings.append(string)
   
    return strings

if __name__ == '__main__':
    args=parse()
    model_man = None
    model_en = None
    print('Current time '+str(datetime.datetime.now()))
    model_man=gensim.models.KeyedVectors.load_word2vec_format('wiki.zh.vec')
    model_en=gensim.models.KeyedVectors.load_word2vec_format('wiki.en.vec')
    print('Current time '+str(datetime.datetime.now()))
    print('finish load models')
    with open(args.text, 'r') as f:
        with open(args.output, 'w') as out:
            for line in f:
                print('-----------------')
                strings=augment(line, model_en, model_man, args.length)
                i=0
                for s in strings:
                  if i == args.number:
                    break 
                  if i == 0:
                    print('Original: '+s)
                  else:
                    print('Synthetic: '+s)
                  out.write(s)
                  i=i+1

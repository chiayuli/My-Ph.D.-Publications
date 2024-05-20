# -*- coding: utf-8 -*-
import datetime
import argparse
import re
import gensim
import string
import random
from google.cloud import translate

def parse():
    ap=argparse.ArgumentParser()
    ap.add_argument('-t', '--text', required=True, help='the CS text')
    #ap.add_argument('-l', '--lang', default='zh', help='english (en) or chinese (zh)')    
    ap.add_argument('-o', '--output', required=True, help='the output file of augmented CS text')    
    ap.add_argument('-l', '--length', default=2, help='how many words in a sentence')
    ap.add_argument('-n', '--number', default=3, help='how many synthetic sentences per line')    
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

def augment(line, model_en, model_man, length, number):
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

    for i in range(int(number)):
      words = re.findall(r'[a-z]+|[\u4e00-\ufaff]+', line, re.UNICODE)
      # there's new alternative English word
      if len(toplist_en) > i:
        if index_en != (len(words) - 1):
          string = ' '.join(words[:index_en]) + ' ' + str(toplist_en[i]) + ' ' + ' '.join(words[index_en+1:]) + '\n'
        else:
          string = ' '.join(words[:index_en]) + ' ' + str(toplist_en[i]) + '\n'
    
      # there's alternative Mandarin word
      if string is not '':
        words = re.findall(r'[a-z]+|[\u4e00-\ufaff]+', string, re.UNICODE)

      if len(toplist_man) > i :
        if index_man != (len(words) -1) :
          string = ' '.join(words[:index_man]) + ' ' + str(toplist_man[i]) + ' ' + ' '.join(words[index_man+1:]) + '\n'
        else:
          string = ' '.join(words[:index_man]) + ' ' + str(toplist_man[i]) + '\n'
      if string != '':
        strings.append(string)

    return strings

def augment2(line, model):
    words = re.findall(r'[a-z]+|[\u4e00-\ufaff]+', line, re.UNICODE)
    strings = [line]
    for w in words:
        if re.match(r'[a-z]+',w):
          if w in model.vocab:
            similar_words = model.similar_by_word(word=w, topn=50)
            for u in similar_words:
              if re.match(r'[\u4e00-\ufaff]+',u[0]):
                print(w)
                print(u[0])
                print("=====")
    return strings
    
def augment3(line, translator):
    words = re.findall(r'[a-z]+|[\u4e00-\ufaff]+', line, re.UNICODE)
    strings = [line]
    if len(words) > 0:
      half = int(len(words)/2)
      if half < 0:
        half = 1
      for i in range(half):
        index = random.randint(1,100) % len(words)
        if re.match(r'[a-z]+', words[index]):
           target='zh-CN'
        else:
           target='en'
        translation = translator.translate(words[index],target_language=target)
        words[index] = translation['translatedText']
      strings.append(' '.join(words)+'\n')
    return strings

if __name__ == '__main__':
    args=parse()
    model_man = None
    model_en = None
    translate_client = translate.Client()
    with open(args.text, 'r') as f:
        with open(args.output, 'w') as out:
            for line in f:
                print('-----------------')
                strings=augment3(line, translate_client)
                for i in range(len(strings)):
                  if i == 0:
                    print('Original: '+strings[i])
                  else:
                    print('Synthetic: '+strings[i])
                  out.write(strings[i])

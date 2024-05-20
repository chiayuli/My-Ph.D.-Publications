# -*- coding: utf-8 -*-
from html.parser import HTMLParser
import datetime
import argparse
import re
import gensim
import string
import random
from google.cloud import translate
import json
def parse():
    ap=argparse.ArgumentParser()
    ap.add_argument('-t', '--text', required=True, help='the CS text')
    ap.add_argument('-o', '--output', required=True, help='the output file of augmented CS text')    
    ap.add_argument('-l', '--length', default=100, help='how many words in a sentence')
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

# good cmi=0.12 and 10 < len(words) < length
def augment3(line, translator, length):
    words = re.findall(r'[a-z]+|[\u4e00-\ufaff]+', line, re.UNICODE)
    strings = [line]
    arr = []
    h = HTMLParser()
    for end in range(int(len(words)/5)+1):
    #for end in range(2):
        cmi=round(random.randint(4,(5+end))*0.1, 2)
        print('cmi '+str(cmi))
        if len(words) < 100:
            # the number of ZH word needs to be replaced by English word
            num = int(round((len(words)*cmi)))
            #print('num '+str(num)+', len(words) '+str(len(words)))
            if num >= len(words):
              num = len(words) - 1
            indexs = random.sample(range(0, len(words)-1), num)
            #print('indexs: '+str(indexs))
            new_words = [w for w in words]
            for idx in indexs:
                if re.match(r'[a-z]+', words[idx]):
                    target='zh-CN'
                else:
                    target='en'
                translation = translator.translate(words[idx],target_language=target)
                new_words[idx] = h.unescape(translation['translatedText']).lower()
        arr.append({'src_zh': line.rstrip().split(), 'src_cs': new_words})
    return arr

if __name__ == '__main__':
    args=parse()
    model_man = None
    model_en = None
    translate_client = translate.Client()
    with open(args.text, 'r') as f:
        with open(args.output, 'w') as out:
            for line in f:
                strings=augment3(line, translate_client, int(args.length))
                for i in range(len(strings)):
                  print(strings[i])
                  out.write(json.dumps(strings[i],ensure_ascii=False)+"\n")

# -*- coding: utf-8 -*-
from html.parser import HTMLParser
import datetime
import argparse
import re
#import gensim
import string
import random
from google.cloud import translate
import json
random.seed(100)
def parse():
    ap=argparse.ArgumentParser()
    ap.add_argument('-t', '--text', required=True, help='the CS text')
    ap.add_argument('-o', '--output', required=True, help='the output file of augmented CS (json)')    
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

# EN/ZH-CMI 2,3,4,5
def augment3(line, translator):
    words = re.findall(r'[a-z]+|[\u4e00-\ufaff]+', line, re.UNICODE)
    strings = [line]
    arr = []
    h = HTMLParser()
    lineSOSEOS = '<sos> '+line+' <eos>'
    zh_total = len(words)
    zh_switches_idx = []

    ## ZH-CMI2-5
    zh_cmi = ((random.randint(0,101) % 4) + 2 ) * 0.1
    num_switches = int(round((1 - zh_cmi) * zh_total, 2)) # the number of ZH word needs to be replaced by English word
    print('[ZH-CMI] '+str(zh_cmi)+ ', [NUM_SWITCHES] '+str(num_switches)+', [TOTAL] '+str(zh_total))
    zh_switches_idx = random.sample(range(1, zh_total), num_switches)
    #print('  [ZH_SWITCH_IDX] '+str(zh_switches_idx))
    new_words = lineSOSEOS.split()
    for idx in zh_switches_idx:
        translation = translator.translate(new_words[idx],target_language='en')
        new_words[idx] = translation['translatedText'].lower()
    arr.append({'src_zh': lineSOSEOS.rstrip().split(), 'src_cs': new_words})
    
    ## EN-CMI2-5
    en_line = translator.translate(line,target_language='en')['translatedText'].lower()
    en_words = re.findall(r'[a-z]+', en_line, re.UNICODE)
    en_total = len(en_words)
    en_lineSOSEOS = '<sos> ' + ' '.join(en_words) + ' <eos>'
    #print('[+++EN_LINE] '+str(en_words))
    print('[+++EN_SOSEOSLINE] '+str(en_lineSOSEOS))
    en_cmi = ((random.randint(0,101) % 4) + 2 ) * 0.1
    num_switches = int(round((1 - en_cmi) * en_total, 2)) # the number of EN word needs to be replaced by ZH word
    print('[EN-CMI] '+str(en_cmi)+ ', [NUM_SWITCHES] '+str(num_switches)+', [EN_TOTAL] '+str(en_total))
    en_switches_idx = random.sample(range(1, en_total), num_switches)
    #print('  [EN_SWITCH_IDX] '+str(en_switches_idx))
    new_words = en_lineSOSEOS.split()
    for idx in en_switches_idx:
        translation = translator.translate(new_words[idx],target_language='zh')
        new_words[idx] = translation['translatedText'].lower()
    arr.append({'src_zh': lineSOSEOS.rstrip().split(), 'src_cs': new_words})

    return arr

if __name__ == '__main__':
    args=parse()
    model_man = None
    model_en = None
    translate_client = translate.Client()
    with open(args.text, 'r') as f:
        with open(args.output, 'w') as out:
            for line in f:
                strings=augment3(line, translate_client)
                for i in range(len(strings)):
                  print(strings[i])
                  out.write(json.dumps(strings[i],ensure_ascii=False)+"\n")

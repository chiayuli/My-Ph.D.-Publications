# -*- coding: utf-8 -*-
import argparse
import re
import string
from google.cloud import translate

def parse():
    ap=argparse.ArgumentParser()
    ap.add_argument('-t', '--text', required=True, help='the CS text')
    ap.add_argument('-o', '--output', required=False, help='the output file of augmented CS text')    
    ap.add_argument('-l', '--length', default=100, help='how many words in a sentence')
    ap.add_argument('-n', '--number', default=3, help='how many synthetic sentences per line')    
    args=ap.parse_args()
    return args

def augment(line, translator, length):
    rmPunc = str.maketrans('', '', string.punctuation)
    translation = translator.translate(line,target_language='en')
    s = translation['translatedText'].lower()
    s = s.translate(rmPunc)
    return s

if __name__ == '__main__':
    args=parse()
    translate_client = translate.Client()
    with open(args.text, 'r') as f:
        for line in f:
            new = augment(line, translate_client, int(args.length))
            print(line.rstrip()+','+new)

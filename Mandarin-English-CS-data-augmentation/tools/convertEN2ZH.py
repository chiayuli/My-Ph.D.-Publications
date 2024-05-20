# -*- coding: utf-8 -*-
import argparse
import re
#from googletrans import Translator
from translate import Translator
def parse():
    ap=argparse.ArgumentParser()
    ap.add_argument('-d','--dictionary',required=True,help='the en-zh dictionary') 
    ap.add_argument('-f','--text',required=True,help='the Code-Mixing text') 
    ap.add_argument('-o','--outfile',required=True,help='the Chinese text') 
    args=ap.parse_args()
    return args

def constructDict(filename):
    # constrctDict
    bidict={}
    with open(filename,'r') as f:
        for line in f:
            w=line.split() # w[0]: english word, w[1]: chinese word
            bidict.update({w[0]:w[1]})
    return bidict

def translate(line, bidict, translator):
    # input: CS sentence
    # return: Chinese sentence
    print(line)
    en_regex=r'[a-z]+'
    zh_regex=r'[\x4e-\xfa]+'
    regex=r'[a-z]+|[\x4e-\xfa]+'
    matches=re.findall(regex,line,re.UNICODE)
    #print(matches)
    string=''
    for m in matches:
        if re.match(en_regex,m):
            if m in bidict:
                print(m+'->'+bidict[m])
                string+=' '+bidict[m]
            else:
                new=translator.translate(m)
                print(m+'->'+new.encode('utf8'))
                string=string+' '+new.encode('utf8')
        else:
           string=string+' '+m
    print(string)
    return string+'\n'

if __name__ == '__main__':
    args=parse()
    bidict=constructDict(args.dictionary)
    translator= Translator(to_lang="zh")
    with open(args.outfile, 'w') as out:
        with open(args.text, 'r') as f:
            for line in f:
                out.write(translate(line, bidict, translator))
        

# -*- coding: utf-8 -*-
import os
import re
import argparse
ap=argparse.ArgumentParser()
ap.add_argument("-f1", "--filename1", required=True, help="name of chinese text file")
ap.add_argument("-f2", "--filename2", required=True, help="name of english text file")
args=vars(ap.parse_args())

def preprocess(line, lang='ZH'):
    #en_regex = r"[0-9]+|[a-zA-Z]+\'*[a-z]*"
    en_regex = r"[a-z]+"
    zh_regex = r"[\u4e00-\ufaff]+"
    #regex = r"[\u4e00-\ufaff]+|[0-9]+|[a-zA-Z<>'-]+"
    #matches = re.findall(regex, line, re.UNICODE)
    words = line.split()
    #prevlang=''
    string=''
   
    if len(words) > 1:
        string = words[0] # the unique id of line 
        if lang == 'ZH':
            for i in range(1, len(words)):
                if re.match(zh_regex, words[i]):
                    string += ' ' + words[i]
                else:
                    print('ZH line contains non-ZH character: %s ' %words[i])
                    return "NaN"
        elif lang == 'EN':
            for i in range(1, len(words)):
                if re.match(en_regex, words[i]):
                    string += ' ' + words[i]
                else:
                    print('EN line contains non-EN character: %s ' %words[i])
                    return "NaN"
        else:
            print('please assign lang to \'ZH\' or \'EN\'')
            return "NaN"
        return string
    else:
        return "NaN"

if __name__ == "__main__":
    # preprocess Chinese text file (remove punctuations and line which contains non-Chinese characters) 
    with open(args["filename1"],'r', encoding="utf-8") as f1:
        with open('chinese_norm.txt', 'w') as n1:
            for line in f1:
                norm_string = preprocess(line, lang='ZH')
                if norm_string != 'NaN':
                    n1.write(norm_string+'\n')
    
    # preprocess English text file (remove punctuations and line which contains non-English words)
    with open(args["filename2"],'r', encoding="utf-8") as f2:
        with open('english_norm.txt', 'w') as n2:
            for line in f2:
                norm_string = preprocess(line, lang='EN')
                if norm_string != 'NaN':
                    n2.write(norm_string+'\n')
    
   

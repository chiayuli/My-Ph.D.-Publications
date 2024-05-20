# python2
# -*- coding: utf-8 -*-
import argparse
import re, os

counter = { 
    "ZH-C1" : 0, 
    "ZH-C2" : 0, 
    "ZH-C3" : 0, 
    "ZH-C4" : 0, 
    "ZH-C5" : 0, 
    "EN-C1" : 0, 
    "EN-C2" : 0, 
    "EN-C3" : 0, 
    "EN-C4" : 0, 
    "EN-C5" : 0
    }

def parser():
    ap=argparse.ArgumentParser(description='compute Code-Mixing Index (CMI)')
    ap.add_argument('-f','--file',required=True,help='the Code-Mixing file')
    ap.add_argument('-o','--output',default='CMI.txt',help='the output file')
    ap.add_argument('-s','--start',default=1,help='the starting point of sentence')
    args=ap.parse_args()
    return args

def computeCMI(line):
    print(line)
    c="NAN"
    en_count=0
    zh_count=0
    en_regex=r'[a-z]+'
    zh_regex=r'[\u4e00-\u9fff]+'
    for m in line:
        if re.match(en_regex,m):
            print('EN '+m)
            en_count+=1
        elif re.match(zh_regex,m):
            print('ZH '+m)
            zh_count+=1
        else:
            print('OTHER '+m)
     
    total=en_count+zh_count
    print('total: '+str(total)+', en_count: '+str(en_count)+', zh_count: '+str(zh_count))
    if total > 0:
        max_count=max(en_count,zh_count)
        cmi=round(100*(1-(float(max_count)/float(total))))
        # find the CMI class for this sentence
        if cmi == 0:
          c="C1"
        elif 0 < cmi and cmi <= 15:
          c="C2"
        elif cmi < 15 and cmi <= 30:
          c="C3"
        elif cmi < 30 and cmi <= 45:
          c="C4"
        else:
          c="C5"
               
        if en_count > zh_count:
            print('cmi '+str(cmi)+' c '+str(c))
            counter["EN-"+str(c)]+=1
            return "EN", round(cmi), "EN-"+str(c)
        else:
            print('cmi '+str(cmi)+' c '+str(c))
            counter["ZH-"+str(c)]+=1
            return "ZH", round(cmi), "ZH-"+str(c)
    else:
        return "Non-speech", -1, str(c)

def normalize(d):
    factor = 1.0 / sum(d.values())
    normalized_d = {k: round(v*factor*100) for k, v in d.items()}
    return normalized_d

if __name__ == "__main__":
    print('hello python')
    args=parser()
    with open(args.output, 'w') as out:
        with open(args.file,'r') as f:
            for line in f:
                if line != '':
                   string = line.split(' ')
                   spk = string[0]
                   dominate, cmi, c = computeCMI(string[int(args.start):])
                   out.write(spk+' '+c+' '+' '.join(string[int(args.start):]))
    print(counter)
    print(normalize(counter))

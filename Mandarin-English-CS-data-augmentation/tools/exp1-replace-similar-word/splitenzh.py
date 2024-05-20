# -*- coding: utf-8 -*-
import os
import re
import argparse
ap=argparse.ArgumentParser()
ap.add_argument("-f", "--filename", required=True, help="name of file")
args=vars(ap.parse_args())

def spliteKeyWord(str):
    #regex = r"[\u4e00-\ufaff]+|[0-9]+|[a-zA-Z]+\'*[a-z]*"
    regex = r"[\u4e00-\ufaff]+|[0-9]+|[a-zA-Z<>'-]+"
    matches = re.findall(regex, str, re.UNICODE)
    string=' '.join(matches)
    return string

if __name__ == "__main__":
    with open(args["filename"],'r', encoding="utf-8") as f:
        for line in f:
            print(spliteKeyWord(line))
    
    

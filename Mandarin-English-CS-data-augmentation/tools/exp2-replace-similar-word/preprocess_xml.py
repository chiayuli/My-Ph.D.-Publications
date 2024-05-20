import xml.etree.ElementTree as ET
import re, sys
root = ET.parse('smsCorpus_zh_2015.03.09.xml').getroot()

def getChinese(context):
    #context = context.decode("utf-8") # convert context from str to unicode
    filtrate = re.compile(u'[^\u4E00-\u9FA5]') # non-Chinese unicode range
    context = filtrate.sub(r'', context) # remove all non-Chinese characters
    #context = context.encode("utf-8") # convert unicode back to str
    return context


with open('smsCorpus_zh.txt', 'w') as f:
    for msg in root.findall('message'):
        line=getChinese(msg.find('text').text)
        print(line)
        f.write(line+'\n')




import argparse
import json
import os
from collections import Counter
from random import random

from tqdm import tqdm


def gen_text():
    counter=json.load(open(freq_source))[:arg.num_class]
    text=open(text_source,encoding='utf-8').read()
    text_test = open(text_test_source, encoding='utf-8').read()
    text_select=''
    text_select_test = ''

    keys=set()
    index={}
    id=0
    for each in counter:
        keys.add(each[0])
    alphabet=list(keys)
    alphabet.sort()
    alphabet=''.join(alphabet)
    with open(alphabet_dest,'w',encoding='utf-8') as f:
        f.write(alphabet)

    print('gen training text...')
    for ch in tqdm(text):
        if ch in keys:
            if index.get(ch):
                index[ch].append(id)
            else:
                index[ch]=[id]
            text_select+=ch
            id+=1


    print('gen testing text...')
    for ch in tqdm(text_test):
        if ch in keys:
            text_select_test+=ch

    freq=Counter(text_select)
    freq=freq.most_common(len(freq))
    json.dump(freq,open(freq_dest,'w',encoding='utf-8'))
    json.dump(index, open(index_dest, 'w', encoding='utf-8'))
    with open(text_dest,'w',encoding='utf-8') as f:
        f.write(text_select)
    with open(text__test_dest,'w',encoding='utf-8') as f:
        f.write(text_select)

    print('training text frequency:')
    print(freq)
    print('training text length:',len(text_select))

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_class',type=int, default=100)
    arg = parser.parse_args()

    if not os.path.exists('data/images/desc'):
        os.makedirs('data/images/desc')
    text_source = 'spider/text.txt'
    freq_source = 'spider/freq.json'
    text_dest = 'data/images/desc/text.txt'
    alphabet_dest = 'data/images/alphabet.txt'
    freq_dest = 'data/images/desc/freq.json'
    index_dest = 'data/images/desc/index.json'

    text_test_source = 'spider/text_test.txt'
    freq_test_source = 'spider/freq_test.json'
    text__test_dest = 'data/images/desc/text_test.txt'
    # freq__test_dest = 'data/images/freq_test_%d.json' % (arg.num_class)

    gen_text()
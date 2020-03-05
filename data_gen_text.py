import argparse
import json
from collections import Counter
from random import random

from tqdm import tqdm


def gen_text(text_source,keys_source,freq_source,text_dest,keys_dest,freq_dest):
    counter=json.load(open(freq_source))[:arg.num_class]
    text=open(text_source,encoding='utf-8').read()
    text_select=''
    keys=set()
    for each in counter:
        keys.add(each[0])
    for ch in tqdm(text):
        if ch in keys:
            # if ch=='的' and random()<0.5:
            #     continue
            text_select+=ch
    freq=Counter(text_select)
    freq=freq.most_common(len(freq))
    keys=list(keys)
    keys.sort()
    keys=''.join(keys)
    json.dump(freq,open(freq_dest,'w',encoding='utf-8'))
    with open(text_dest,'w',encoding='utf-8') as f:
        f.write(text_select)
    with open(keys_dest,'w',encoding='utf-8') as f:
        f.write(keys)
    print('frequency:')
    print(freq)
    print('text length:',len(text_select))

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_class',type=int, default=10)
    arg = parser.parse_args()

    print('gen trainning and validation data...')
    text_source = 'spider/text.txt'
    keys_source = 'spider/keys.txt'
    freq_source = 'spider/freq.json'
    text_dest = 'data/images/text_%d.txt' % (arg.num_class)
    keys_dest = 'data/images/keys_%d.txt' % (arg.num_class)
    freq_dest = 'data/images/freq_text_%d.json' % (arg.num_class)
    gen_text(text_source, keys_source, freq_source, text_dest, keys_dest, freq_dest)

    print('gen testing data...')
    text_source = 'spider/text_test.txt'
    keys_source = 'spider/keys_test.txt'
    freq_source = 'spider/freq_test.json'
    text_dest = 'data/images/text_test%d.txt' % (arg.num_class)
    keys_dest = 'data/images/keys_test%d.txt' % (arg.num_class)
    freq_dest = 'data/images/freq_text_test%d.json' % (arg.num_class)
    gen_text(text_source,keys_source,freq_source,text_dest,keys_dest,freq_dest)
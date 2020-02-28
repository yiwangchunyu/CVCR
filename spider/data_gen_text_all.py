import argparse
import json
import os
from collections import Counter

from tqdm import tqdm


def gen_text_file_all():
    print('generating text file all from spider/data...')
    text=''
    dirs = os.listdir('data')
    for dir in tqdm(dirs):
        files = os.listdir(os.path.join('data', dir))
        for file in files:
            with open(os.path.join('data', dir, file), encoding='utf-8') as f:
                text += f.read()
    keys = set()
    for ch in text:
        keys.add(ch)
    keys=list(keys)
    keys.sort()
    keys=''.join(keys)
    c=Counter(text)
    freq=c.most_common(len(c))
    json.dump(freq,open('freq.json','w',encoding='utf-8'))
    with open('text.txt','w',encoding='utf-8') as f:
        f.write(text)
    with open('keys.txt','w',encoding='utf-8') as f:
        f.write(keys)
    print('text length:',len(text))
    print('keys size:', len(keys))
    print('data saved at:','spider/text.txt','spider/keys.txt')
    print('charator frequency save at spider/dreq.json')

def gen_text_file_all_test():
    print('generating text file all from spider/data_test...')
    text=''
    dirs = os.listdir('data_test')
    for dir in tqdm(dirs):
        files = os.listdir(os.path.join('data_test', dir))
        for file in files:
            with open(os.path.join('data_test', dir, file), encoding='utf-8') as f:
                text += f.read()
    keys = set()
    for ch in text:
        keys.add(ch)
    keys=list(keys)
    keys.sort()
    keys=''.join(keys)
    c=Counter(text)
    freq=c.most_common(len(c))
    json.dump(freq,open('freq_test.json','w',encoding='utf-8'))
    with open('text_test.txt','w',encoding='utf-8') as f:
        f.write(text)
    with open('keys_test.txt','w',encoding='utf-8') as f:
        f.write(keys)
    print('text length:',len(text))
    print('keys size:', len(keys))
    print('data saved at:','spider/text_test.txt','spider/keys_test.txt')
    print('charator frequency save at spider/freq_test.json')


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, default='test', help='')
    arg = parser.parse_args()
    if arg.type=='train':
        gen_text_file_all()
    elif arg.type=='test':
        gen_text_file_all_test()
    else:
        print('input type[train|test]!!!!!!!!!')
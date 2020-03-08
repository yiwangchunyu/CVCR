'''
V2:中文横版数据生成（简体，繁体），完成一小步，接下来竖版
'''
import argparse
import json
import os
import random
import shutil
from collections import Counter

import numpy as np
from PIL import Image, ImageFont, ImageDraw, ImageFilter
from tqdm import tqdm

from compute_stds_means import compute_std_mean
from utils import trans_by_zhtools


def sampleWord(length=10,source='',type=1,ch=''):

    word=''
    with open(source,'r',encoding='utf-8') as f:
        all=f.read()
        if type==1:
            # drop=0
            #training data
            ids=index[ch]
            id=random.choice(ids)
            pos_in_word=random.randint(0,length-1)
            word=ch
            pos_pre=id-1
            pos_aft=id+1
            while len(word)<=pos_in_word and pos_pre>=0:
                if dataSaver.counter_train[all[pos_pre]]<arg.basenum_train:
                    word=all[pos_pre]+word
                # else:
                #     drop+=1
                pos_pre-=1
            while len(word)<length and pos_aft<len(all):
                if dataSaver.counter_train[all[pos_aft]]<arg.basenum_train:
                    word=word+all[pos_aft]
                # else:
                #     drop += 1
                pos_aft+=1
            # print(drop,word,ch,pos_in_word)
        elif type==2:
            #validation data
            ids=index[ch]
            id=random.choice(ids)
            pos_in_word=random.randint(0,length)
            word=ch
            pos_pre=id-1
            pos_aft=id+1
            while len(word)<pos_in_word and pos_pre>=0:
                if dataSaver.counter_valid[all[pos_pre]]<arg.basenum_valid:
                    word=all[pos_pre]+word
                pos_pre-=1
            while len(word)<length and pos_aft<len(all):
                if dataSaver.counter_valid[all[pos_aft]]<arg.basenum_valid:
                    word=word+all[pos_aft]
                pos_aft+=1

        elif type==3:
            start=random.randint(0,len(all)-length)
            end = start + length

            word = all[start:end]
        else:
            print('error: sample word error, type:',type)
    if len(word)!=length:
        # print()
        print('error: sample word error,',ch,pos_in_word,word,len(word))
        # exit(-1)
    return word

def createAnImageR90(backgroundPath,w,h):
    backgrounds = os.listdir(backgroundPath)
    background = random.choice(backgrounds)
    backgroundImg = Image.open(os.path.join(backgroundPath,background))
    # backgroundImg.show()
    backgroundImg=backgroundImg.transpose(Image.ROTATE_90)
    # backgroundImg.show()
    x, y = random.randint(0, backgroundImg.size[0] - w), random.randint(0, backgroundImg.size[1] - h)
    backgroundImg = backgroundImg.crop((x, y, x + w, y + h))
    return backgroundImg

def createAnImage(backgroundPath,w,h):
    backgrounds = os.listdir(backgroundPath)
    background = random.choice(backgrounds)
    backgroundImg = Image.open(os.path.join(backgroundPath,background))
    x, y = random.randint(0, backgroundImg.size[0] - w), random.randint(0, backgroundImg.size[1] - h)
    backgroundImg = backgroundImg.crop((x, y, x + w, y + h))
    return backgroundImg

def sampleFontSizeR90():
    font_size = random.randint(20, 27)

    return font_size

def sampleFontSize():
    font_size = random.randint(24, 27)

    return font_size


def sampleFont(fontRoot):
    fonts=os.listdir(fontRoot)
    font=random.choice(fonts)
    return fontRoot+'/'+font


def sampleWordColor():
    font_color_choice = [[54, 54, 54], [54, 54, 54], [105, 105, 105]]
    font_color = random.choice(font_color_choice)

    noise = np.array([random.randint(0, 10), random.randint(0, 10), random.randint(0, 10)])
    font_color = (np.array(font_color) + noise).tolist()

    # print('font_color：',font_color)

    return tuple(font_color)


def randomXYR90(size, font_size, hsum):
    width, height = size
    # print(bground_size)
    # 为防止文字溢出图片，x，y要预留宽高
    # x = random.randint(0, width - font_size * 10)
    # y = random.randint(0, int((height - font_size) / 4))
    x = random.randint(0, width - font_size)
    y = random.randint(0, height - hsum)
    return x, y

def randomXY(size, font_size):
    width, height = size
    # print(bground_size)
    # 为防止文字溢出图片，x，y要预留宽高
    x = random.randint(0, width - font_size * 10)
    y = random.randint(0, int((height - font_size) / 4))
    return x, y

def darkenFunc(image):
    # .SMOOTH
    # .SMOOTH_MORE
    # .GaussianBlur(radius=2 or 1)
    # .MedianFilter(size=3)
    # 随机选取模糊参数
    filter_ = random.choice(
        [ImageFilter.SMOOTH,
         ImageFilter.SMOOTH_MORE,
         ImageFilter.GaussianBlur(radius=1.3)]
    )
    image = image.filter(filter_)
    # image = img.resize((290,32))

    return image


def gen(ch='', type=1, length=10,source=None):
    # 随机选取10个字符
    random_word=sampleWord(source=source,type=type, ch=ch)

    trans_word=random_word
    if arg.trans:
        trans_word=trans_by_zhtools(random_word)
    # print(random_word)
    # print(traditional_word)
    if arg.direction=='vertical':
        # 生成一张背景图片，已经剪裁好，宽高为32*280
        raw_image = createAnImageR90(arg.backgroundRoot, 32, 280)

        # 随机选取字体
        font_name = sampleFont(arg.fontRoot)
        # 随机选取字体颜色
        font_color = sampleWordColor()

        # print(font_name)

        #计算长度宽度
        contain=0
        while contain==0:
            # 随机选取字体大小
            font_size = sampleFontSizeR90()
            font = ImageFont.truetype(font_name, font_size)
            hsum=0
            for ch in trans_word:
                w,h=font.getsize(ch)
                hsum+=h
            if hsum<280:
                contain=1
        # 随机选取文字贴合的坐标 x,y
        draw_x, draw_y = randomXYR90(raw_image.size, font_size,hsum)
        # 将文本贴到背景图片
        draw = ImageDraw.Draw(raw_image)
        pos=draw_y
        for ch in trans_word:
            draw.text((draw_x, pos), ch, fill=font_color, font=font)
            pos+=font.getsize(ch)[1]


    elif arg.direction=='horizontal':
        # 生成一张背景图片，已经剪裁好，宽高为32*280
        raw_image = createAnImage(arg.backgroundRoot, 280, 32)
        # 随机选取字体大小
        font_size = sampleFontSize()
        # 随机选取字体
        font_name = sampleFont(arg.fontRoot)
        # 随机选取字体颜色
        font_color = sampleWordColor()
        # 随机选取文字贴合的坐标 x,y
        draw_x, draw_y = randomXY(raw_image.size, font_size)
        # 将文本贴到背景图片
        # print(font_name)
        font = ImageFont.truetype(font_name, font_size)
        draw = ImageDraw.Draw(raw_image)
        draw.text((draw_x, draw_y), trans_word, fill=font_color, font=font)

    else:
        print('error: pelese select direction in arg!')
        exit(-1)
    # 随机选取作用函数和数量作用于图片
    # random_choice_in_process_func()
    raw_image = darkenFunc(raw_image)
    # raw_image = raw_image.rotate(0.3)
    # 保存文本信息和对应图片名称
    # with open(save_path[:-1]+'.txt', 'a+', encoding='utf-8') as file:
    # file.write('10val/' + str(num) + '.png ' + random_word + '\n')

    dataSaver.save(raw_image,random_word,type)
    return random_word

class DataSaver():
    def __init__(self,trainRoot,validRoot,testRoot,trainLabelPath,validLabelPath,testLabelPath):
        self.train_id = 0
        self.valid_id = 0
        self.test_id = 0
        self.trainRoot=trainRoot
        self.validRoot=validRoot
        self.testRoot = testRoot
        self.trainLabelPath = trainLabelPath
        self.validLabelPath = validLabelPath
        self.testLabelPath = testLabelPath
        self.train_labels=[]
        self.valid_labels = []
        self.test_labels = []
        self.word_list=''
        self.counter_train=Counter()
        self.counter_valid = Counter()
        self.counter_test = Counter()
        pass

    def save(self,img,label,type=1):

        if type==1:
            fname='train_%d.png' % (self.train_id)
            img.save(os.path.join(self.trainRoot,fname))
            self.train_labels.append(('%s/%s' % (self.trainRoot, fname), label))
            self.train_id += 1
            self.counter_train.update(label)

        elif type==2:
            # 保存为验证样本
            fname = 'valid_%d.png' % (self.valid_id)
            img.save(os.path.join(self.validRoot, fname))
            self.valid_labels.append(('%s/%s' % (self.validRoot, fname), label))
            self.valid_id += 1
            self.counter_valid.update(label)
        else:
            fname = 'test_%d.png' % (self.test_id)
            img.save(os.path.join(self.testRoot, fname))
            self.test_labels.append(('%s/%s' % (self.testRoot, fname), label))
            self.test_id += 1
            self.counter_test.update(label)

    def finish(self):
        json.dump(self.valid_labels,open(self.validLabelPath,'w'))
        json.dump(self.train_labels, open(self.trainLabelPath, 'w'))
        json.dump(self.test_labels, open(self.testLabelPath, 'w'))
        #数据集字符统计
        with open(statistic_dest,'w',encoding='utf-8') as f:
            print('training data:')
            f.write('training data:\n')
            print('charator counter:')
            f.write('charator counter:\n')
            print(self.counter_train)
            f.write(self.counter_train.__str__()+'\n')
            print('num_samples:',self.train_id)
            f.write('num_samples: %d\n'%(self.train_id))
            print('-------------------------------------------------------')
            f.write('-------------------------------------------------------\n')
            print('validation data:')
            f.write('validation data:\n')
            print('charator counter:')
            f.write('charator counter:\n')
            print(self.counter_valid)
            f.write(self.counter_valid.__str__()+'\n')
            print('num_samples:', self.valid_id)
            f.write('num_samples: %d\n'%(self.valid_id))
            print('-------------------------------------------------------')
            f.write('-------------------------------------------------------\n')
            print('testing data:')
            f.write('testing data:\n')
            print('charator counter:')
            f.write('charator counter:\n')
            print(self.counter_test)
            f.write(self.counter_test.__str__()+'\n')
            print('num_samples:', self.test_id)
            f.write('num_samples: %d\n'%(self.test_id))
            print('-------------------------------------------------------')
            f.write('-------------------------------------------------------\n')
            #数据mean，std
            compute_std_mean(arg.trainLabelPath, imgW=280, imgH=32, rgb=True)

def main():
    print('deleting files...',arg.trainRoot,arg.validRoot)
    if not os.path.exists(arg.trainRoot):
        os.mkdir(arg.trainRoot)
    for file in os.listdir(arg.trainRoot):
        os.remove(os.path.join(arg.trainRoot,file))
    if not os.path.exists(arg.validRoot):
        os.mkdir(arg.validRoot)
    for file in os.listdir(arg.validRoot):
        os.remove(os.path.join(arg.validRoot,file))
    if not os.path.exists(arg.testRoot):
        os.mkdir(arg.testRoot)
    for file in os.listdir(arg.testRoot):
        os.remove(os.path.join(arg.testRoot,file))
    print('deleting files...down')

    print('gen training data...')
    for each in tqdm(freq):
        ch=each[0]
        while dataSaver.counter_train[ch]<arg.basenum_train:
            gen(ch,type=1,source=text_source)

    print('gen validation data...')
    for each in tqdm(freq):
        ch = each[0]
        while dataSaver.counter_valid[ch]<arg.basenum_valid:
            gen(ch,type=2,source=text_source)

    print('gen testing data...')
    for i in tqdm(range(arg.num_test)):
        gen(type=3,source=text_test_source)

    dataSaver.finish()
    return


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--trans', type=bool, default=True, help='transform to traditinal chinese charactor')
    parser.add_argument('--blc', type=bool, default=True, help='')
    # parser.add_argument('--direction', type=str, default='vertical', help='')
    parser.add_argument('--direction', type=str, default='horizontal', help='')
    # parser.add_argument('--num_class', type=int, default=10, help='')
    parser.add_argument('--basenum_train', type=int, default=50, help='')
    parser.add_argument('--basenum_valid', type=int, default=20, help='')
    parser.add_argument('--num_test', type=int, default=1000, help='')
    parser.add_argument('--trainRoot', type=str, default='data/images/train', help='')
    parser.add_argument('--validRoot', type=str, default='data/images/valid', help='')
    parser.add_argument('--testRoot', type=str, default='data/images/test', help='')
    parser.add_argument('--trainLabelPath', type=str, default='data/images/train_label.txt', help='')
    parser.add_argument('--validLabelPath', type=str, default='data/images/valid_label.txt', help='')
    parser.add_argument('--testLabelPath', type=str, default='data/images/test_label.txt', help='')
    parser.add_argument('--backgroundRoot', type=str, default='data/background', help='')
    parser.add_argument('--fontRoot', type=str, default='data/fonts', help='')
    arg = parser.parse_args()

    alphabet_source = 'data/images/alphabet.txt'
    text_source='data/images/desc/text.txt'
    text_test_source = 'data/images/desc/text_test.txt'
    freq_source = 'data/images/desc/freq.json'
    index_source = 'data/images/desc/index.json'
    statistic_dest='data/images/desc/statistcis.txt'

    # freq_dest = 'data/images/freq.json'
    dataSaver = DataSaver(arg.trainRoot, arg.validRoot, arg.testRoot,arg.trainLabelPath, arg.validLabelPath,arg.testLabelPath)

    with open(alphabet_source,encoding='utf-8') as f:
        alphabet=f.read()
    num_class=len(alphabet)
    freq=json.load(open(freq_source,encoding='utf-8'))
    freq.reverse()
    print('text source freq reverse:')
    print(freq)
    index=json.load(open(index_source,encoding='utf-8'))

    main()
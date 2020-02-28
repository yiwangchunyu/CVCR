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

from utils import trans_by_zhtools


def sampleWord(length=10,source=''):

    word=''
    with open(source,'r',encoding='utf-8') as f:
        all=f.read()
        if len(all)<length-1:
            return word
        try:
            start = random.randint(0,len(all)-length-1)
        except Exception as e:
            print('randint failed in function sampleWord',e)
            return word
        end = start+length
        word=all[start:end]
    return word

def createAnImage(backgroundPath,w,h):
    backgrounds = os.listdir(backgroundPath)
    background = random.choice(backgrounds)
    backgroundImg = Image.open(os.path.join(backgroundPath,background))
    # backgroundImg.show()
    backgroundImg=backgroundImg.transpose(Image.ROTATE_90)
    # backgroundImg.show()
    x, y = random.randint(0, backgroundImg.size[0] - w), random.randint(0, backgroundImg.size[1] - h)
    backgroundImg = backgroundImg.crop((x, y, x + w, y + h))
    return backgroundImg


def sampleFontSize():
    font_size = random.randint(20, 27)

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


def randomXY(size, font_size, hsum):
    width, height = size
    # print(bground_size)
    # 为防止文字溢出图片，x，y要预留宽高
    # x = random.randint(0, width - font_size * 10)
    # y = random.randint(0, int((height - font_size) / 4))
    x = random.randint(0, width - font_size)
    y = random.randint(0, height - hsum)
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


def gen(type=1):
    # 随机选取10个字符
    random_word=''
    if type==1:
        random_word=sampleWord(10,text_source)
    elif type==2:
        random_word = sampleWord(10,text_source)
        while dataSaver.is_in_keys(random_word) is False:
            random_word = sampleWord(10,text_source)
    elif type==3:
        random_word = sampleWord(10, text_test_source)
        while dataSaver.is_in_keys(random_word) is False:
            random_word = sampleWord(10, text_test_source)
    else:
        print('error: in function gen() at datagenV2!')

    if len(random_word)<10:
        print('warning: len(random_word<10)!!!')
        return

    if arg.trans:
        random_word=trans_by_zhtools(random_word)
    # 生成一张背景图片，已经剪裁好，宽高为32*280
    raw_image = createAnImage(arg.backgroundRoot, 32, 280)

    # 随机选取字体
    font_name = sampleFont(arg.fontRoot)
    # 随机选取字体颜色
    font_color = sampleWordColor()

    # print(font_name)

    #计算长度宽度
    contain=0
    while contain==0:
        # 随机选取字体大小
        font_size = sampleFontSize()
        font = ImageFont.truetype(font_name, font_size)
        hsum=0
        for ch in random_word:
            w,h=font.getsize(ch)
            hsum+=h
        if hsum<280:
            contain=1
    # 随机选取文字贴合的坐标 x,y
    draw_x, draw_y = randomXY(raw_image.size, font_size,hsum)
    # 将文本贴到背景图片
    draw = ImageDraw.Draw(raw_image)
    pos=draw_y
    for ch in random_word:
        draw.text((draw_x, pos), ch, fill=font_color, font=font)
        pos+=font.getsize(ch)[1]
    # 随机选取作用函数和数量作用于图片
    # random_choice_in_process_func()
    raw_image = darkenFunc(raw_image)
    # raw_image = raw_image.rotate(0.3)
    # 保存文本信息和对应图片名称
    # with open(save_path[:-1]+'.txt', 'a+', encoding='utf-8') as file:
    # file.write('10val/' + str(num) + '.png ' + random_word + '\n')

    dataSaver.save(raw_image,random_word,type)

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
        self.keys=set()
        self.word_list=''
        pass

    def save(self,img,label,type=1):

        if type==1:
            fname='train_%d.png' % (self.train_id)
            img.save(os.path.join(self.trainRoot,fname))
            self.train_labels.append(('%s/%s' % (self.trainRoot, fname), label))
            self.train_id += 1
            for ch in label:
                self.keys.add(ch)
            self.word_list += label
        elif type==2:
            # 保存为验证样本
            fname = 'valid_%d.png' % (self.valid_id)
            img.save(os.path.join(self.validRoot, fname))
            self.valid_labels.append(('%s/%s' % (self.validRoot, fname), label))
            self.valid_id += 1
            self.word_list+=label
        else:
            fname = 'test_%d.png' % (self.test_id)
            img.save(os.path.join(self.testRoot, fname))
            self.test_labels.append(('%s/%s' % (self.testRoot, fname), label))
            self.test_id += 1


    def is_in_keys(self,word):
        st=set(word)
        res=st.issubset(self.keys)
        return res

    def finish(self):
        keys=list(self.keys)
        keys.sort()
        keys=''.join(keys)
        print(keys)
        print('length:',len(keys))
        with open(alphabet_dest,'w',encoding='utf-8') as f:
            f.write(keys)
        json.dump(self.valid_labels,open(self.validLabelPath,'w'))
        json.dump(self.train_labels, open(self.trainLabelPath, 'w'))
        json.dump(self.test_labels, open(self.testLabelPath, 'w'))
        #数据集字符统计
        counter=Counter(self.word_list)
        counter=counter.most_common(len(counter))
        print('frequency:')
        print(counter)

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

    threshold=int((arg.num_samples/10)*9)
    for i in tqdm(range(arg.num_samples)):
        if i==0:
            print('gen training data...')
        if i == threshold:
            print('gen testing data...')
        if i<threshold:
            gen(type=1)
        else:
            gen(type=2)
    print('gen testing data....')
    for i in tqdm(range(int(arg.num_samples*3/10))):
        gen(type=3)
    dataSaver.finish()
    return


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--trans', type=bool, default=False, help='transform to traditinal chinese charactor')
    parser.add_argument('--num_class', type=int, default=10, help='')
    parser.add_argument('--num_samples', type=int, default=100, help='')
    parser.add_argument('--trainRoot', type=str, default='data/images/trainV3', help='')
    parser.add_argument('--validRoot', type=str, default='data/images/validV3', help='')
    parser.add_argument('--testRoot', type=str, default='data/images/testV3', help='')
    parser.add_argument('--trainLabelPath', type=str, default='data/images/trainV3label.txt', help='')
    parser.add_argument('--validLabelPath', type=str, default='data/images/validV3label.txt', help='')
    parser.add_argument('--testLabelPath', type=str, default='data/images/testV3label.txt', help='')
    parser.add_argument('--backgroundRoot', type=str, default='data/background', help='')
    parser.add_argument('--fontRoot', type=str, default='data/fonts', help='')
    arg = parser.parse_args()
    text_source='data/images/text_%d.txt'%(arg.num_class)
    text_test_source = 'data/images/text_test%d.txt' % (arg.num_class)
    alphabet_dest = 'data/images/alphabet.txt'
    dataSaver = DataSaver(arg.trainRoot, arg.validRoot, arg.testRoot,arg.trainLabelPath, arg.validLabelPath,arg.testLabelPath)
    main()
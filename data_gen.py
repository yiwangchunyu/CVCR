import json
import os
import random
import numpy as np
import h5py
import lmdb
import six
from PIL import ImageFont, Image, ImageDraw, ImageFilter

MAX_PADDING=5
FONT_SIZE=(15,30)
FONT_PATH='data/fonts'
TEXT_LEN=(2,12)

class MyGaussianBlur(ImageFilter.Filter):
    name = "GaussianBlur"

    def __init__(self, radius=2, bounds=None):
        self.radius = radius
        self.bounds = bounds

    def filter(self, image):
        if self.bounds:
            clips = image.crop(self.bounds).gaussian_blur(self.radius)
            image.paste(clips, self.bounds)
            return image
        else:
            return image.gaussian_blur(self.radius)

def gaussiaNoise(image):
    img = image.astype(np.int16)  # 此步是为了避免像素点小于0，大于255的情况
    mu = 0
    sigma = 10
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            for k in range(img.shape[2]):
                img[i, j, k] = img[i, j, k] + random.gauss(mu=mu, sigma=sigma)
    img[img > 255] = 255
    img[img < 0] = 0
    img = img.astype(np.uint8)
    return img


def saltPepperNoise(img, proportion=0.05):
    noise_img =img.copy()
    height,width =noise_img.shape[0],noise_img.shape[1]
    num = int(height*width*proportion)#多少个像素点添加椒盐噪声
    for i in range(num):
        w = random.randint(0,width-1)
        h = random.randint(0,height-1)
        if random.randint(0,1) ==0:
            noise_img[h,w] = np.zeros(noise_img.shape[2])
        else:
            noise_img[h,w] = np.zeros(noise_img.shape[2])+255
    return noise_img

class Font():
    def __init__(self):
        pass

    def sample(self,size):
        font = ImageFont.truetype(FONT_PATH + "/simkai.ttf", size, encoding="utf-8")
        return font

class Img():
    def __init__(self):

        pass
    def sample(self):
        pass

class Text():
    def __init__(self,num_class):
        self.fid=0
        self.circle=0
        self.ch_id=0
        self.num_class = num_class
        self.text_fnames = []

        self.keys = json.load(open('data/text/keys_%d.json'%(num_class)))
        self.text_path='data/text/text_%d'%(num_class)
        self.text_dirs=os.listdir(self.text_path)
        for d in self.text_dirs:
            fnames = os.listdir(os.path.join(self.text_path,d))
            fnames = [os.path.join(self.text_path,d,name) for name in fnames]
            self.text_fnames.extend(fnames)

    def getChar(self):
        with open(self.text_fnames[self.fid]) as f:
            all=f.read()
            res = all[self.ch_id]
        self.ch_id+=1
        if self.ch_id>=len(all):
            self.ch_id=0
            self.fid+=1
            if self.fid>=len(self.text_fnames):
                self.fid=0
                self.circle+=1
        return res,self.keys['ch2id'][res]

    def sample_len(self):
        len = random.randint(TEXT_LEN[0],TEXT_LEN[1])
        return len

    def sample_padding(self):
        return random.randint(0,MAX_PADDING)


class DataManager():
    def __init__(self,num_class,type):
        self.dpath='data/dataset_%d_%s.h5' % (num_class,type)
        self.num_class=num_class
        self.f = h5py.File(self.dpath,'w')
        self.f.attrs['num_class']=num_class
        self.f.attrs['type'] = type
        self.id=0

    def add(self,img, text_labels):
        d = self.f.create_dataset(str(self.id),data=img)
        d.attrs['text_labels']=text_labels
        self.id+=1

class LmdbDataManager():
    def __init__(self,num_class,type):
        self.keys=set()
        self.train_path='data/dataset_train_%d_%s' % (num_class,type)
        self.test_path = 'data/dataset_test_%d_%s' % (num_class, type)
        self.num_class=num_class

        # self.train_lmdb=lmdb.open(self.train_path, map_size=1073741824)
        self.train_lmdb = lmdb.open(self.train_path)
        self.train_writer=self.train_lmdb.begin(write=True)
        self.train_writer.put('num_class'.encode(),str(num_class).encode())
        self.train_writer.put('type'.encode(), str(type).encode())
        self.train_id=0

        # self.test_lmdb = lmdb.open(self.test_path, map_size=1073741824)
        self.test_lmdb = lmdb.open(self.test_path)
        self.test_writer = self.test_lmdb.begin(write=True)
        self.test_writer.put('num_class'.encode(), str(num_class).encode())
        self.test_writer.put('type'.encode(), str(type).encode())
        self.test_id = 0

    def add(self,img, text):
        for ch in text:
            self.keys.add(ch)
        rdm=random.random()
        bf=six.BytesIO()
        img.save(bf,'PNG')

        if rdm<=0.3:
            img_key = 'image-%d' % self.test_id
            label_key = 'label-%d' % self.test_id
            self.test_writer.put(img_key.encode(), bf.getvalue())
            self.test_writer.put(label_key.encode(), text.encode())
            # self.test_writer.commit()
            self.test_id+=1
        else:
            img_key = 'image-%d' % self.train_id
            label_key = 'label-%d' % self.train_id
            self.train_writer.put(img_key.encode(), bf.getvalue())
            self.train_writer.put(label_key.encode(), text.encode())
            # self.train_writer.commit()
            self.train_id += 1

    def close(self):
        self.train_writer.put('num_samples'.encode(), str(self.train_id).encode())
        self.train_writer.commit()
        self.train_lmdb.close()
        self.test_writer.put('num_samples'.encode(), str(self.test_id).encode())
        self.test_writer.commit()
        self.test_lmdb.close()
        keys=list(self.keys)
        keys.sort()
        json.dump(keys,open('data/keys_%d.json'%(len(keys)),'w'))

def main(num_class=20,type='',circle=1):
    text = Text(num_class)
    # dataManager = DataManager(num_class,type=type)
    lmdbdataManager = LmdbDataManager(num_class,type=type)
    cur_circle=0
    data_size=0
    while cur_circle<circle:
        padding = text.sample_padding()
        len=text.sample_len()
        font_size=random.randint(FONT_SIZE[0],FONT_SIZE[1])
        font=Font().sample(size=font_size)
        txt=''
        labels=[]
        pos=[]
        img_w,img_h=0,padding
        for i in range(len):
            ch,label=text.getChar()
            txt+=ch
            labels.append(label)
            ch_w, ch_h = font.getsize(ch)
            pos.append((padding, img_h))
            img_w=max(img_w,ch_h+2*padding)
            img_h+=ch_h+padding
        img = Image.new('RGB',size=(img_w,img_h))
        draw=ImageDraw.Draw(img)
        for i,p in enumerate(pos):
            draw.text(p,txt[i],fill=(255,255,255),font=font)
        # img.show()
        # img=img.filter(MyGaussianBlur(radius=1))
        img=Image.fromarray(gaussiaNoise(np.array(img)))
        img = Image.fromarray(saltPepperNoise(np.array(img),proportion=0.005))
        # img.show()
        # dataManager.add(img,labels)
        lmdbdataManager.add(img,txt)
        data_size+=1
        if cur_circle<text.circle:
            print("circle:",text.circle)
            print("datasize:", data_size)
        cur_circle=text.circle
    lmdbdataManager.close()

if __name__=="__main__":
    main(circle=100,type='gray')
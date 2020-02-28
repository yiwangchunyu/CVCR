import sys
from io import BytesIO

import h5py
import lmdb
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class hdh5Dataset(Dataset):

    def __init__(self, path=None, transform=None):
        self.f=h5py.File(path)
        self.transform = transform
        self.fkeys=list(self.f.keys())
        self.len=len(self.fkeys)
        self.num_class=self.f.attrs['num_class']

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        obj=self.f[self.fkeys[index]]
        img=obj[:]
        labels=obj.attrs['text_labels']
        if self.transform is not None:
            img = self.transform(img)

        return (img, labels)

class lmdbDataset(Dataset):
    def __init__(self, path=None, transform=None):
        self.env = lmdb.open(
            path,
            max_readers=4,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False
            )
        if not self.env:
            print('cannot create lmdb from %s' % (path))
            sys.exit(0)

        with self.env.begin(write=False) as txn:
            num_samples = int(txn.get('num_samples'.encode()))
            self.num_samples = num_samples
        self.transform = transform

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        assert index < self.num_samples, 'index range error'
        with self.env.begin(write=False) as txn:
            img_key = 'image-%d' % index
            label_key = 'label-%d' % index
            img=Image.open(BytesIO(txn.get(img_key.encode())))
            label=np.frombuffer(txn.get(label_key.encode()),dtype=int)

        if self.transform is not None:
            img = self.transform(img)

        return (img, label)

class resizeNormalize(object):

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    def __call__(self, img):
        # img=Image.fromarray(img)
        img=img.transpose(Image.ROTATE_90) # 逆时针旋转270
        img = img.resize(self.size, self.interpolation)
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        return img


class alignCollate(object):

    def __init__(self, imgH=32, imgW=100, keep_ratio=False, min_ratio=1):
        self.imgH = imgH
        self.imgW = imgW
        self.keep_ratio = keep_ratio
        self.min_ratio = min_ratio

    def __call__(self, batch):
        images, labels = zip(*batch)

        imgH = self.imgH
        imgW = self.imgW
        if self.keep_ratio:
            ratios = []
            for image in images:
                w, h = image.size
                ratios.append(w / float(h))
            ratios.sort()
            max_ratio = ratios[-1]
            imgW = int(np.floor(max_ratio * imgH))
            imgW = max(imgH * self.min_ratio, imgW)  # assure imgH >= imgW

        transform = resizeNormalize((imgW, imgH))
        images = [transform(image) for image in images]
        images = torch.cat([t.unsqueeze(0) for t in images], 0)

        return images, labels
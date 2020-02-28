import json

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from compute_stds_means import compute_std_mean


class DatasetV2(Dataset):
    def __init__(self,datalabelsPath,transform=None,imgW=160,imgH=32,mean=None,std=None):
        super(DatasetV2, self).__init__()
        self.datalabels=json.load(open(datalabelsPath))
        self.length=len(self.datalabels)
        self.transform=transform
        self.imgW=imgW
        self.imgH = imgH
        self.mean=mean
        self.std=std

    def __len__(self):
        return  self.length

    def __getitem__(self, index):
        image_path,label=self.datalabels[index]

        if self.transform is not None:
            image = Image.open(image_path)
            image=self.transform(image)
        else:
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            h, w = image.shape
            image = cv2.resize(image, (0, 0), fx=self.imgW / w, fy=self.imgH / h, interpolation=cv2.INTER_CUBIC)
            image = (np.reshape(image, (32, self.imgW, 1))).transpose(2, 0, 1)
            image = image.astype(np.float32) / 255.
            image = torch.from_numpy(image).type(torch.FloatTensor)
            image.sub_(self.mean).div_(self.std)

        return image,label

class resizeNormalize(object):

    def __init__(self, size, mean=0.588,std=0.193, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation
        self.mean=mean
        self.std=std
        self.toTensor = transforms.ToTensor()

    def __call__(self, img):
        # img=Image.fromarray(img)
        # img=img.transpose(Image.ROTATE_90) # 逆时针旋转270
        img=img.convert('L')
        img = img.resize(self.size, self.interpolation)
        img = self.toTensor(img)
        img=img.div_(255)
        img.sub_(self.mean).div_(self.std)

        return img

if __name__=="__main__":
    mean, std = compute_std_mean('data/trainV2label.txt', (160, 32))
    dataset=DatasetV2('data/trainV2label.txt',mean=mean,std=std)
    dataloader=DataLoader(dataset, batch_size=8, shuffle=False)
    print(dataset.__len__())
    for i_batch, (image, label) in enumerate(dataloader):
        print(image.shape)
        pass
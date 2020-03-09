import json

import cv2
import numpy as np
import torch
import torchvision
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from compute_stds_means import compute_std_mean


class Dataset(Dataset):
    def __init__(self,datalabelsPath,nc=1,transform=None, target_transform=None):
        super(Dataset, self).__init__()
        self.datalabels=json.load(open(datalabelsPath))
        self.length=len(self.datalabels)
        if nc==1:
            self.mode='L'
        elif nc==3:
            self.mode='RGB'
        self.transform=transform
        self.target_transform = target_transform


    def __len__(self):
        return  self.length

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'

        img_path = self.datalabels[index][0]
        try:
            img = Image.open(img_path).convert('L')
        except IOError:
            print('Corrupted image for %d' % index)
            return self[index + 1]

        if self.transform is not None:
            img = self.transform(img)

        label = self.datalabels[index][1]

        if self.target_transform is not None:
            label = self.target_transform(label)

        return (img, label)

class ResizeNormalize(object):

    def __init__(self, img_width, img_height, mean_std_file=None, direction='horizontal', interpolation=Image.BILINEAR):
        self.img_width = img_width
        self.img_height = img_height
        if mean_std_file is not None:
            self.mean_std = json.load(open(mean_std_file))
            self.mean = self.mean_std['mean']
            self.std = self.mean_std['std']
        else:
            self.mean = 0.5
            self.std = 0.5

        self.direction = direction
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()
        self.transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            # torchvision.transforms.Normalize(mean=self.mean, std=self.std)
        ])

    def __call__(self, img):
        if self.direction == 'vertical':
            img.transpose(Image.ROTATE_90)
        img = img.resize((self.img_width,self.img_height), self.interpolation)
        img = self.transforms(img)
        img.sub_(self.mean).div_(self.std)
        return img


class AlignCollate(object):

    def __init__(self, img_height=32, img_width=100, mean_std_file=None):
        self.img_height = img_height
        self.img_width = img_width
        self.transform = ResizeNormalize(img_width=self.img_width, img_height=self.img_height, direction='horizontal')

    def __call__(self, batch):
        images, labels = zip(*batch)

        images = [self.transform(image) for image in images]
        images = torch.cat([t.unsqueeze(0) for t in images], 0)

        return images, labels

if __name__=="__main__":
    dataset=Dataset('data/images/train_label.txt')
    dataloader=DataLoader(dataset,
                          batch_size=8,
                          shuffle=False,
                          collate_fn=AlignCollate()
                          )
    print(dataset.__len__())
    for i_batch, (image, label) in enumerate(dataloader):
        print(image.shape)
        pass
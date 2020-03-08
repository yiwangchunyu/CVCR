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
    def __init__(self,datalabelsPath,transform=None, target_transform=None):
        super(Dataset, self).__init__()
        self.datalabels=json.load(open(datalabelsPath))
        self.length=len(self.datalabels)
        self.transform=transform
        self.target_transform = target_transform


    def __len__(self):
        return  self.length

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'

        img_path = self.datalabels[index][0]
        try:
            img = Image.open(img_path).convert('RGB')
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

    def __init__(self, img_width, img_height, mean_std_file, direction='horizontal'):
        self.img_width = img_width
        self.img_height = img_height
        self.mean_std = json.load(open(mean_std_file))
        self.mean = self.mean_std['mean']
        self.std = self.mean_std['std']
        self.direction = direction

        self.transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=self.mean, std=self.std)
        ])

    def __call__(self, img):
        if self.direction == 'vertical':
            img.transpose(Image.ROTATE_90)
        img = np.array(img)
        h, w, c = img.shape
        height = self.img_height
        width = int(w * height / h)
        if width >= self.img_width:
            img = cv2.resize(img, (self.img_width, self.img_height))
        else:
            img = cv2.resize(img, (width, height))
            img_pad = np.zeros((self.img_height, self.img_width, c), dtype=img.dtype)
            img_pad[:height, :width, :] = img
            img = img_pad
        img = np.asarray(img)
        img = self.transforms(img)
        return img


class AlignCollate(object):

    def __init__(self, img_height=32, img_width=100, mean_std_file='data/images/desc/mean_std.json'):
        self.img_height = img_height
        self.img_width = img_width
        self.transform = ResizeNormalize(img_width=self.img_width, img_height=self.img_height, mean_std_file=mean_std_file, direction='horizontal')

    def __call__(self, batch):
        images, labels = zip(*batch)

        images = [self.transform(image) for image in images]
        images = torch.cat([t.unsqueeze(0) for t in images], 0)

        return images, labels

if __name__=="__main__":
    dataset=Dataset('data/trainV2label.txt')
    dataloader=DataLoader(dataset, batch_size=8, shuffle=False)
    print(dataset.__len__())
    for i_batch, (image, label) in enumerate(dataloader):
        print(image.shape)
        pass
import argparse
import json
import os
import random

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm


def compute_std_mean(path, imgW,imgH, rgb=False):
    print('computing mean and std...')
    datalabels=json.load(open(path))
    mean = 0
    std = 0
    if rgb:
        pass
    else:
        mean=0
        std=0
        num_images=len(datalabels)
        for datalabel in tqdm(datalabels):
            image = cv2.imread(datalabel[0])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            h, w = image.shape
            image = cv2.resize(image, (0, 0), fx=imgW / w, fy=imgH / h, interpolation=cv2.INTER_CUBIC)
            image=np.array(image)/255
            mean+=image[:,:].mean()
            std+=image[:,:].std()
        mean=mean/num_images
        std=std/num_images
    print('mean=',mean,'std=',std)
    json.dump({'mean':mean,'std':std},open('data/dataV2_mean_std.json','w'))
    return mean, std


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--rgb', type=bool,default=False, help='')
    parser.add_argument('--path', type=str, default='data/trainV2label.txt', help='')
    parser.add_argument('--imgH', type=int, default=32, help='the height of the input image to network')
    parser.add_argument('--imgW', type=int, default=160, help='the width of the input image to network')
    arg = parser.parse_args()
    compute_std_mean(arg.path, (arg.imgW,arg.imgH), arg.rgb)
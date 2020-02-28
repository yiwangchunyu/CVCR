import argparse
import os
import random

import numpy as np
import torch
from torch import optim
from torch.autograd import Variable
from torch.backends import cudnn
from torch.utils.data import DataLoader

import utils
from compute_stds_means import compute_std_mean
from datasetV2 import DatasetV2, resizeNormalize
from models.crnn import CRNN

# custom weights initialization called on crnn
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def backward_hook(self, grad_input, grad_output):
    for g in grad_input:
        g[g != g] = 0   # replace all nan/inf in gradients to zero


def train(crnn, train_loader, criterion, epoch):
    for p in crnn.parameters():
        p.requires_grad = True
    crnn.train()
    loss_avg = utils.averager()
    for i_batch, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        preds = crnn(images)
        batch_size = images.size(0)
        text, length = converter.encode(labels)
        # print(converter.decode(text,length))
        preds_size = torch.IntTensor([preds.size(0)] * batch_size)

        cost = criterion(preds, text, preds_size, length) / batch_size
        crnn.zero_grad()
        cost.backward()
        optimizer.step()
        loss_avg.add(cost)

        if (i_batch+1) % arg.displayInterval == 0:
            print('[%d/%d][%d/%d] Loss: %f' %
                  (epoch, arg.nepoch, i_batch, len(train_loader), loss_avg.val()))
            loss_avg.reset()


def valid(crnn, valid_loader, criterion, epoch, max_i):
    print('Start val')
    for p in crnn.parameters():
        p.requires_grad = False
    crnn.eval()
    i = 0
    tag=1
    n_correct = 0
    n_count = 0
    loss_avg = utils.averager()

    for i_batch, (images, labels) in enumerate(valid_loader):
        images = images.to(device)
        preds = crnn(images)
        batch_size = images.size(0)
        n_count+=batch_size
        text, length = converter.encode(labels)
        preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))
        cost = criterion(preds, text, preds_size, length) / batch_size
        loss_avg.add(cost)
        _, preds = preds.max(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)
        sim_preds = converter.decode(preds.data, preds_size.data, raw=False)
        for pred, target in zip(sim_preds, labels):
            if pred == target:
                n_correct += 1

        if tag==1:
            raw_preds = converter.decode(preds.data, preds_size.data, raw=True)[:10]
            for raw_pred, pred, gt in zip(raw_preds, sim_preds, labels):
                print('%-20s => %-20s, gt: %-20s' % (raw_pred, pred, gt))
            tag=0

        if (i_batch + 1) % arg.displayInterval == 0:
            print('[%d/%d][%d/%d]' %
                  (epoch, arg.nepoch, i_batch, len(valid_loader)))

        if i_batch == max_i:
            break


    print(n_correct)
    print(n_count)
    accuracy = n_correct / float(n_count)
    print('Test loss: %f, accuray: %f' % (loss_avg.val(), accuracy))

    return accuracy


def main(crnn, train_loader, valid_loader, criterion, optimizer):

    crnn = crnn.to(device)
    criterion = criterion.to(device)

    best_accuracy = arg.best_acc
    for epoch in range(arg.nepoch):
        train(crnn, train_loader, criterion, epoch)
        accuracy = valid(crnn, valid_loader, criterion, epoch, max_i=1000)
        for p in crnn.parameters():
            p.requires_grad = True
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(crnn.state_dict(), '{0}/crnnV2_savepoint_{1}_{2}.pth'.format(arg.expr, epoch, accuracy))
            torch.save(crnn.state_dict(), '{0}/crnnV2_best.pth'.format(arg.expr))
            print('model saved, best acc=%f'%(best_accuracy))
        print("epoch: {0}, accuracy: {1}".format(epoch, accuracy))
    print('finished, best acc=%f' % (best_accuracy))
    return


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_root', default='data/trainV2label.txt',help='path to training dataset')
    parser.add_argument('--valid_root', default='data/validV2label.txt', help='path to testing dataset')
    parser.add_argument('--alphabet', default='data/alphabet.txt', help='')
    parser.add_argument('--num_workers', type=int, default=2, help='number of data loading workers')
    parser.add_argument('--batch_size', type=int, default=8, help='input batch size')
    parser.add_argument('--imgH', type=int, default=32, help='the height of the input image to network')
    parser.add_argument('--imgW', type=int, default=160, help='the width of the input image to network')
    parser.add_argument('--nepoch', type=int, default=25, help='number of epochs to train for')
    parser.add_argument('--cuda', action='store_true', default=False,help='enables cuda')
    parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
    parser.add_argument('--pretrained', default='', help="path to pretrained model (to continue training)")
    parser.add_argument('--opt', default='adadelta', help='select optimizer')
    parser.add_argument('--type', default='gray', help='select type')
    parser.add_argument('--expr', default='expr', help='Where to store samples and models')
    parser.add_argument('--displayInterval', type=int, default=5, help='Interval to be displayed')
    parser.add_argument('--testInterval', type=int, default=400, help='Interval to be displayed')
    parser.add_argument('--saveInterval', type=int, default=200, help='Interval to save model')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate for Critic, not used by adadealta')
    parser.add_argument('--mean', type=float, default=0.588, help='')
    parser.add_argument('--std', type=float, default=0.192, help='')
    parser.add_argument('--best_acc', type=float, default=0.5, help='')
    parser.add_argument('--keep_ratio', action='store_true', default=False,help='whether to keep ratio for image resize')
    arg = parser.parse_args()

    params={}
    with open(arg.alphabet,encoding='utf-8') as f:
        alphabets=f.read()
    params['alphabets']=alphabets
    params['num_class'] = len(alphabets)


    manualSeed = 10
    random.seed(manualSeed)
    np.random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    cudnn.benchmark = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # store model path
    if not os.path.exists('./expr'):
        os.mkdir('./expr')
    mean,std=compute_std_mean(arg.train_root,imgW=arg.imgW,imgH=arg.imgH)

    train_dataset = DatasetV2(
        arg.train_root,
        imgH=arg.imgH,
        imgW=arg.imgW,
        mean=mean,
        std=std
    )
    valid_dataset = DatasetV2(
        arg.valid_root,
        imgH=arg.imgH,
        imgW=arg.imgW,
        mean=mean,
        std=std
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=arg.batch_size,
        num_workers=arg.num_workers,
        shuffle=True
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=arg.batch_size,
        num_workers=arg.num_workers,
        shuffle=True
    )
    nc = 1
    num_class = params['num_class'] + 1
    converter = utils.ConverterV2(params['alphabets'])
    criterion = torch.nn.CTCLoss(reduction='sum')
    crnn = CRNN(32, nc, num_class, 256)
    crnn.apply(weights_init)

    # setup optimizer
    if arg.opt == 'adam':
        optimizer = optim.Adam(crnn.parameters(), lr=arg.lr,
                               betas=(params.beta1, 0.999))
    elif arg.opt == 'adadelta':
        optimizer = optim.Adadelta(crnn.parameters())
    else:
        optimizer = optim.RMSprop(crnn.parameters(), lr=arg.lr)

    crnn.register_backward_hook(backward_hook)

    main(crnn, train_loader, valid_loader, criterion, optimizer)
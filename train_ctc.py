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
import dataset_ctc as dataset
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


def train(crnn, train_loader, criterion, epoch, plot):
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
            plot.add_loss(loss_avg.val())
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
    plot= utils.Plot(nepoch=arg.nepoch, fname='data/loss.png')
    best_accuracy = arg.best_acc
    for epoch in range(arg.nepoch):
        train(crnn, train_loader, criterion, epoch, plot)
        accuracy = valid(crnn, valid_loader, criterion, epoch, max_i=1000)
        for p in crnn.parameters():
            p.requires_grad = True
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(crnn.state_dict(), '{0}/crnnV2_savepoint_{1}_{2}.pth'.format(arg.expr, epoch, accuracy))
            torch.save(crnn.state_dict(), '{0}/crnnV2_best.pth'.format(arg.expr))
            print('model saved, best acc=%f'%(best_accuracy))
        print("epoch: {0}, accuracy: {1}".format(epoch, accuracy))
        plot.add_acc(accuracy,epoch)
    print('finished, best acc=%f' % (best_accuracy))
    plot.show()
    return


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_root', default='data/images/train_label.txt',help='path to training dataset')
    parser.add_argument('--valid_root', default='data/images/valid_label.txt', help='path to testing dataset')
    parser.add_argument('--alphabet', default='data/images/alphabet.txt', help='')
    parser.add_argument('--num_workers', type=int, default=2, help='number of data loading workers')
    parser.add_argument('--batch_size', type=int, default=8, help='input batch size')
    parser.add_argument('--imgH', type=int, default=32, help='the height of the input image to network')
    parser.add_argument('--imgW', type=int, default=160, help='the width of the input image to network')
    parser.add_argument('--nepoch', type=int, default=25, help='number of epochs to train for')
    parser.add_argument('--cuda', action='store_true', default=False,help='enables cuda')
    parser.add_argument('--opt', default='adam', help='select optimizer')
    parser.add_argument('--nc', type=int, default=3, help='')
    parser.add_argument('--expr', default='expr', help='Where to store samples and models')
    parser.add_argument('--displayInterval', type=int, default=5, help='Interval to be displayed')
    parser.add_argument('--testInterval', type=int, default=400, help='Interval to be displayed')
    parser.add_argument('--saveInterval', type=int, default=200, help='Interval to save model')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate for Critic, not used by adadealta')
    parser.add_argument('--best_acc', type=float, default=0.5, help='')
    parser.add_argument('--keep_ratio', action='store_true', default=False,help='whether to keep ratio for image resize')
    parser.add_argument('--mean_std_file', type=str, default='data/images/desc/mean_std.json', help='')
    arg = parser.parse_args()

    params={}
    with open(arg.alphabet,encoding='utf-8') as f:
        alphabets=f.read()
    params['alphabets']=alphabets
    num_class = len(alphabets)+1


    manualSeed = 10
    random.seed(manualSeed)
    np.random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    cudnn.benchmark = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # store model path
    if not os.path.exists('./expr'):
        os.mkdir('./expr')

    train_dataset = dataset.Dataset(arg.train_root)
    valid_dataset = dataset.Dataset(arg.valid_root,
                                    transform=dataset.ResizeNormalize(img_width=arg.imgW,
                                                                      img_height=arg.imgH,
                                                                      mean_std_file=arg.mean_std_file)
                                    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=arg.batch_size,
        num_workers=arg.num_workers,
        shuffle=True,
        collate_fn=dataset.AlignCollate(img_height=arg.imgH, img_width=arg.imgW, mean_std_file=arg.mean_std_file)
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=1,
        num_workers=arg.num_workers,
        shuffle=True
    )

    converter = utils.ConverterV2(params['alphabets'])
    criterion = torch.nn.CTCLoss(reduction='sum')
    crnn = CRNN(arg.imgH, arg.nc, num_class, 256)
    print(crnn)
    crnn.apply(weights_init)

    # setup optimizer
    if arg.opt == 'adam':
        optimizer = optim.Adam(crnn.parameters(), lr=arg.lr,
                               betas=(0.9, 0.999))
    elif arg.opt == 'adadelta':
        optimizer = optim.Adadelta(crnn.parameters())
    else:
        optimizer = optim.RMSprop(crnn.parameters(), lr=arg.lr)

    crnn.register_backward_hook(backward_hook)

    main(crnn, train_loader, valid_loader, criterion, optimizer)
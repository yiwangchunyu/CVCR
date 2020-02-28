import argparse

import torch

from torch import optim
from torch.autograd import Variable
from torch.nn import CTCLoss
from torch.utils.data import DataLoader

import utils
from models import crnn
import dataset
from models.crnn import CRNN


def test(arg, net, test_dataset, criterion, image, text, length, max_iter=100):
    print('test start...')
    converter = utils.Converter(arg.num_class)
    for p in net.parameters():
        p.requires_grad = False
    net.eval()
    data_loader = torch.utils.data.DataLoader(
        test_dataset,
        shuffle=True,
        batch_size=arg.batch_size,
        num_workers=arg.num_workers,
        collate_fn=dataset.alignCollate(imgH=arg.imgH, imgW=arg.imgW, keep_ratio=arg.keep_ratio),
    )
    test_iter = iter(data_loader)

    i = 0
    n_correct = 0
    test_loss_avg = utils.averager()

    max_iter = min(max_iter, len(data_loader))
    for i in range(max_iter):
        data = test_iter.next()
        i += 1
        cpu_images, cpu_texts = data
        batch_size = cpu_images.size(0)
        utils.loadData(image, cpu_images)
        t, l = converter.encode(cpu_texts)
        utils.loadData(text, t)
        utils.loadData(length, l)

        preds = crnn(image)
        preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))
        cost = criterion(preds, text, preds_size, length) / batch_size
        test_loss_avg.add(cost)

        _, preds = preds.max(2)
        # preds = preds.squeeze(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)
        sim_preds = converter.decode(preds, preds_size)
        for pred, target in zip(sim_preds, cpu_texts):
            if pred.size == target.size and (pred == target).all():
                n_correct += 1

    # raw_preds = converter.decode(preds.data, preds_size)[:10]
    # for raw_pred, pred, gt in zip(raw_preds, sim_preds, cpu_texts):
    #     print('%-20s => %-20s, gt: %-20s' % (raw_pred, pred, gt))

    accuracy = n_correct / float(max_iter * arg.batch_size)
    print('Test loss: %f, accuray: %f' % (test_loss_avg.val(), accuracy))

def main(arg):
    print(arg)
    train_dataset=dataset.lmdbDataset(
        path=arg.train_root,
        # transform=dataset.resizeNormalize((imgW,imgH)),
    )
    test_dataset = dataset.lmdbDataset(
        path=arg.test_root,
        # transform=dataset.resizeNormalize((arg.imgW,arg.imgH)),
    )
    d=test_dataset.__getitem__(0)
    l=test_dataset.__len__()
    train_loader = DataLoader(
        train_dataset,
        num_workers=arg.num_workers,
        batch_size=arg.batch_size,
        collate_fn=dataset.alignCollate(imgH=arg.imgH, imgW=arg.imgW, keep_ratio=arg.keep_ratio),
        shuffle=True,
        drop_last=True)

    criterion = CTCLoss()
    converter= utils.Converter(arg.num_class)
    crnn = CRNN(imgH=arg.imgH, nc=3, nclass=arg.num_class+1, nh=256)

    # custom weights initialization called on crnn
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

    crnn.apply(weights_init)
    print(crnn)

    image = torch.FloatTensor(arg.batch_size, 3, arg.imgH, arg.imgW)
    text = torch.IntTensor(arg.batch_size * 5)
    length = torch.IntTensor(arg.batch_size)

    image = Variable(image)
    text = Variable(text)
    length = Variable(length)


    # loss averager
    loss_avg = utils.averager()

    # setup optimizer
    if arg.opt=='adam':
        optimizer = optim.Adam(crnn.parameters(), 0.01,
                               betas=(0.5, 0.999))
    elif arg.opt=='adadelta':
        optimizer = optim.Adadelta(crnn.parameters())
    else:
        optimizer = optim.RMSprop(crnn.parameters(), 0.01)

    for epoch in range(arg.n_epoch):
        train_iter = iter(train_loader)
        i = 0
        while i < len(train_loader):
            for p in crnn.parameters():
                p.requires_grad = True
            crnn.train()

            data = train_iter.next()
            cpu_images, cpu_texts = data
            batch_size = cpu_images.size(0)
            utils.loadData(image, cpu_images)
            text_labels, l = converter.encode(cpu_texts)
            utils.loadData(text, text_labels)
            utils.loadData(length, l)

            preds = crnn(image)
            preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))
            cost = criterion(preds, text, preds_size, length) / batch_size
            crnn.zero_grad()
            cost.backward()
            optimizer.step()

            loss_avg.add(cost)
            i += 1

            if i % arg.displayInterval == 0:
                print('[%d/%d][%d/%d] Loss: %f' %
                      (epoch, arg.n_epoch, i, len(train_loader), loss_avg.val()))
                loss_avg.reset()

            if i % arg.testInterval == 0:
                test(arg, crnn, test_dataset, criterion,image, text, length)

            # do checkpointing
            if i % arg.saveInterval == 0:
                name = '{0}/netCRNN_{1}_{2}_{3}_{4}.pth'.format(arg.model_dir, arg.num_class, arg.type,epoch, i)
                torch.save(
                    crnn.state_dict(), name)
                print('model saved at ',name)
    torch.save(
        crnn.state_dict(), '{0}/netCRNN_{1}_{2}.pth'.format(arg.model_dir, arg.num_class, arg.type))

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_class', type=int, default=20,help='num_class')
    parser.add_argument('--train_root', default='data/dataset_train_20_gray',help='path to training dataset')
    parser.add_argument('--test_root', default='data/dataset_test_20_gray', help='path to testing dataset')
    parser.add_argument('--num_workers', type=int, default=0, help='number of data loading workers')
    parser.add_argument('--batch_size', type=int, default=4, help='input batch size')
    parser.add_argument('--imgH', type=int, default=32, help='the height of the input image to network')
    parser.add_argument('--imgW', type=int, default=100, help='the width of the input image to network')
    parser.add_argument('--n_epoch', type=int, default=25, help='number of epochs to train for')
    parser.add_argument('--cuda', action='store_true', default=False,help='enables cuda')
    parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
    parser.add_argument('--pretrained', default='', help="path to pretrained model (to continue training)")
    parser.add_argument('--opt', default='', help='select optimizer')
    parser.add_argument('--type', default='gray', help='select type')
    parser.add_argument('--model_dir', default='models', help='Where to store samples and models')
    parser.add_argument('--displayInterval', type=int, default=10, help='Interval to be displayed')
    parser.add_argument('--testInterval', type=int, default=400, help='Interval to be displayed')
    parser.add_argument('--saveInterval', type=int, default=200, help='Interval to save model')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate for Critic, not used by adadealta')
    parser.add_argument('--keep_ratio', action='store_true', default=False,help='whether to keep ratio for image resize')
    arg = parser.parse_args()
    main(arg)
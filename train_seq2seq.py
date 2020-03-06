import argparse
import random
import os

import numpy as np

import torch
import torchvision
import torch.backends.cudnn as cudnn
import torch.utils.data

import utils_seq2seq as utils
import dataset_seq2seq as dataset

import models.seq2seq as crnn
from utils import Plot

cudnn.benchmark = True

parser = argparse.ArgumentParser()
parser.add_argument('--train_root', default='data/images/train_label.txt', help='path to training dataset')
parser.add_argument('--valid_root', default='data/images/valid_label.txt', help='path to testing dataset')
parser.add_argument('--alphabet', default='data/images/alphabet.txt', help='')
parser.add_argument('--num_workers', type=int, default=4, help='number of data loading num_workers')
parser.add_argument('--batch_size', type=int, default=16, help='input batch size')
parser.add_argument('--imgH', type=int, default=32, help='the height of the input image to network')
parser.add_argument('--imgW', type=int, default=280, help='the width of the input image to network')
parser.add_argument('--hidden_size', type=int, default=256, help='size of the lstm hidden state')
parser.add_argument('--nepoch', type=int, default=2, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate for Critic, default=0.00005')
parser.add_argument('--encoder', type=str, default='', help="path to encoder (to continue training)")
parser.add_argument('--decoder', type=str, default='', help='path to decoder (to continue training)')
parser.add_argument('--expr', default='./expr/', help='Where to store samples and models')
parser.add_argument('--random_sample', default=True, action='store_true',
                    help='whether to sample the dataset with random sampler')
parser.add_argument('--teaching_forcing_prob', type=float, default=0.5, help='where to use teach forcing')
parser.add_argument('--max_width', type=int, default=71, help='the width of the feature map out from cnn')
parser.add_argument('--displayInterval', type=int, default=2, help='Interval to be displayed')
parser.add_argument('--validInterval', type=int, default=1, help='Interval to be displayed')
parser.add_argument('--mean_std_file', type=str, default='data/images/desc/mean_std.json', help='whether use gpu')

arg = parser.parse_args()
print(arg)


def train(image, text, encoder, decoder, criterion, train_loader, valid_loader,teach_forcing_prob=1):
    # optimizer
    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=arg.lr, betas=(0.5, 0.999))
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=arg.lr, betas=(0.5, 0.999))

    # loss averager
    loss_avg = utils.Averager()
    best_acc=0
    for epoch in range(arg.nepoch):
        train_iter = iter(train_loader)

        for i in range(len(train_loader)):
            cpu_images, cpu_texts = train_iter.next()
            batch_size = cpu_images.size(0)

            for encoder_param, decoder_param in zip(encoder.parameters(), decoder.parameters()):
                encoder_param.requires_grad = True
                decoder_param.requires_grad = True
            encoder.train()
            decoder.train()

            target_variable = converter.encode(cpu_texts)
            utils.load_data(image, cpu_images)

            # CNN + BiLSTM
            encoder_outputs = encoder(image)
            target_variable = target_variable.to(device)
            # start decoder for SOS_TOKEN
            decoder_input = target_variable[utils.SOS_TOKEN].to(device)
            decoder_hidden = decoder.initHidden(batch_size).to(device)

            loss = 0.0
            teach_forcing = True if random.random() > teach_forcing_prob else False
            if teach_forcing:
                for di in range(1, target_variable.shape[0]):
                    decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden,
                                                                                encoder_outputs)
                    loss += criterion(decoder_output, target_variable[di])
                    decoder_input = target_variable[di]
            else:
                for di in range(1, target_variable.shape[0]):
                    decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden,
                                                                                encoder_outputs)
                    loss += criterion(decoder_output, target_variable[di])
                    topv, topi = decoder_output.data.topk(1)
                    ni = topi.squeeze()
                    decoder_input = ni
            encoder.zero_grad()
            decoder.zero_grad()
            loss.backward()
            encoder_optimizer.step()
            decoder_optimizer.step()

            loss_avg.add(loss)

            if (i+1) % arg.displayInterval == 0:
                print('[Epoch {0:0>4}/{1:0>4}] [Batch {2:0>4}/{3:0>4}] Loss: {4}'.format(epoch, arg.nepoch, i, len(train_loader),
                                                                         loss_avg.val()))
                plot.add_loss(float(loss_avg.val()))
                loss_avg.reset()

        if (epoch+1)%arg.validInterval==0:
            accuracy = evaluate(image, text, encoder, decoder, valid_loader, max_eval_iter=100)
            # print('validation acc:',accuracy)
            plot.add_acc(accuracy,epoch+1)
            if accuracy>best_acc:
                best_acc=accuracy
                print('best acc:',accuracy,', model saved')
                torch.save(encoder.state_dict(), '{0}/encoder_best.pth'.format(arg.expr))
                torch.save(decoder.state_dict(), '{0}/decoder_best.pth'.format(arg.expr))
        # save checkpoint
        # torch.save(encoder.state_dict(), '{0}/encoder_{1}.pth'.format(arg.expr, epoch))
        # torch.save(decoder.state_dict(), '{0}/decoder_{1}.pth'.format(arg.expr, epoch))
    print('best acc:',best_acc)
    plot.show()


def evaluate(image, text, encoder, decoder, data_loader, max_eval_iter=100):
    for e, d in zip(encoder.parameters(), decoder.parameters()):
        e.requires_grad = False
        d.requires_grad = False

    encoder.eval()
    decoder.eval()
    val_iter = iter(data_loader)

    n_correct = 0
    n_total = 0
    loss_avg = utils.Averager()

    for i in range(min(len(data_loader), max_eval_iter)):
        cpu_images, cpu_texts = val_iter.next()
        batch_size = cpu_images.size(0)
        utils.load_data(image, cpu_images)

        target_variable = converter.encode(cpu_texts)
        n_total += len(cpu_texts[0]) + 1

        decoded_words = []
        decoded_label = []
        encoder_outputs = encoder(image)
        target_variable = target_variable.to(device)
        decoder_input = target_variable[0].to(device)
        decoder_hidden = decoder.initHidden(batch_size).to(device)

        for di in range(1, target_variable.shape[0]):
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.data.topk(1)
            ni = topi.squeeze(1)
            decoder_input = ni
            if ni == utils.EOS_TOKEN:
                decoded_label.append(utils.EOS_TOKEN)
                break
            else:
                decoded_words.append(converter.decode(ni))
                decoded_label.append(ni)

        for pred, target in zip(decoded_label, target_variable[1:, :]):
            if pred == target:
                n_correct += 1

        if i % 10 == 0:
            texts = cpu_texts[0]
            print('pred: {}, gt: {}'.format(''.join(decoded_words), texts))

    accuracy = n_correct / float(n_total)
    print('{0}/{1}'.format(n_correct,n_total))
    print('Test loss: {}, accuray: {}'.format(loss_avg.val(), accuracy))

    # for e, d in zip(encoder.parameters(), decoder.parameters()):
    #     e.requires_grad = True
    #     d.requires_grad = True
    #
    # encoder.train()
    # decoder.train()

    return accuracy

def main():
    if not os.path.exists(arg.expr):
        os.makedirs(arg.expr)

    # create train dataset
    train_dataset = dataset.TextLineDataset(data_path=arg.train_root, transform=None)
    sampler = dataset.RandomSequentialSampler(train_dataset, arg.batch_size)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=arg.batch_size,
        shuffle=False,
        drop_last=True,
        sampler=sampler,
        num_workers=int(arg.num_workers),
        collate_fn=dataset.AlignCollate(img_height=arg.imgH, img_width=arg.imgW,mean_std_file=arg.mean_std_file))

    # create test dataset
    valid_dataset = dataset.TextLineDataset(data_path=arg.valid_root,
                                           transform=dataset.ResizeNormalize(img_width=arg.imgW,
                                                                             img_height=arg.imgH,
                                                                             mean_std_file=arg.mean_std_file))
    valid_loader = torch.utils.data.DataLoader(valid_dataset,
                                              shuffle=True,
                                              batch_size=1,
                                              num_workers=int(arg.num_workers))

    # create crnn/seq2seq/attention network
    encoder = crnn.Encoder(channel_size=3, hidden_size=arg.hidden_size)
    # for prediction of an indefinite long sequence
    decoder = crnn.Decoder(hidden_size=arg.hidden_size, output_size=num_class, dropout_p=0.1,
                           max_length=arg.max_width)
    print(encoder)
    print(decoder)
    encoder.apply(utils.weights_init)
    decoder.apply(utils.weights_init)
    if arg.encoder:
        print('loading pretrained encoder model from %s' % arg.encoder)
        encoder.load_state_dict(torch.load(arg.encoder))
    if arg.decoder:
        print('loading pretrained encoder model from %s' % arg.decoder)
        decoder.load_state_dict(torch.load(arg.decoder))

    # create input tensor
    image = torch.FloatTensor(arg.batch_size, 3, arg.imgH, arg.imgW)
    text = torch.LongTensor(arg.batch_size)

    criterion = torch.nn.NLLLoss()

    encoder.to(device)
    decoder.to(device)
    image = image.to(device)
    text = text.to(device)
    criterion = criterion.to(device)

    # train crnn
    train(image, text, encoder, decoder, criterion, train_loader, valid_loader,teach_forcing_prob=arg.teaching_forcing_prob)

    # do evaluation after training
    # evaluate(image, text, encoder, decoder, test_loader, max_eval_iter=100)


if __name__ == "__main__":
    # load alphabet
    with open(arg.alphabet, encoding='utf-8') as f:
        alphabet = f.read()

    # define convert bwteen string and label index
    converter = utils.ConvertBetweenStringAndLabel(alphabet)

    # len(alphabet) + SOS_TOKEN + EOS_TOKEN
    num_class = len(alphabet) + 2

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    plot=Plot(arg.nepoch)
    main()
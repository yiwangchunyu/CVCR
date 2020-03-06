import json

import numpy as np
# import opencc
import torch
from torch.autograd import Variable
import langconv
import matplotlib.pyplot as plt
# def trans_by_opencc(word):
# #     #将简体转换成繁体
# #     # cc = opencc.OpenCC('s2t')
# #     cc = opencc.OpenCC('mix2t')
# #     return cc.convert(word)

def trans_by_zhtools(word):
    # 将简体转换成繁体
    word = langconv.Converter('zh-hant').convert(word)
    return word


class LabelTransformer():
    def __init__(self):
        pass

class averager(object):
    """Compute average for `torch.Variable` and `torch.Tensor`. """

    def __init__(self):
        self.reset()

    def add(self, v):
        if isinstance(v, Variable):
            count = v.data.numel()
            v = v.data.sum()
        elif isinstance(v, torch.Tensor):
            count = v.numel()
            v = v.sum()

        self.n_count += count
        self.sum += v

    def reset(self):
        self.n_count = 0
        self.sum = 0

    def val(self):
        res = 0
        if self.n_count != 0:
            res = self.sum / float(self.n_count)
        return res


def loadData(v, data):
    v.resize_(data.size()).copy_(data)

class Converter():
    def __init__(self,num_class):
        self.keys=json.load(open('data/text/keys_%d.json'%(num_class)))

    def encode(self,text_labels):
        length = [len(text_label) for text_label in text_labels]
        t=[]
        for label in text_labels:
            t.extend(label)
        t=np.array(t).flatten()+1
        t=torch.IntTensor(t)
        l=torch.IntTensor(length)
        return t, l

    def decode(self,text_labels, lengths):
        assert text_labels.numel() == lengths.sum(), "texts with length: {} does not match declared length: {}".format(text_labels.numel(),
                                                                                                            lengths.sum())
        t = []
        index = 0
        for i in range(lengths.numel()):
            l = lengths[i]
            char_list = []
            for j in range(index,index + l):
                if text_labels[j] != 0 and (not (j > index and text_labels[j - 1] == text_labels[j])):
                    char_list.append(text_labels[j]-1)
            t.append(char_list)
            index += l
        return np.array(t)

    def raw_label2text(self,labels):
        res=''
        for label in labels:
            if label==0:
                res+='-'
            else:
                res+=self.keys['id2ch'][label-1]
        return res

    def label2text(self,labels):
        res=''
        for label in labels:
            res+=self.keys['id2ch'][label-1]
        return res

class ConverterV2():
    """Convert between str and label.

        NOTE:
            Insert `blank` to the alphabet for CTC.

        Args:
            alphabet (str): set of the possible characters.
            ignore_case (bool, default=True): whether or not to ignore all of the case.
        """

    def __init__(self, alphabet):

        self.alphabet = alphabet + '-'  # for `-1` index

        self.dict = {}
        for i, char in enumerate(alphabet):
            # NOTE: 0 is reserved for 'blank' required by wrap_ctc
            self.dict[char] = i + 1

    def encode(self, text):
        """Support batch or single str.

        Args:
            text (str or list of str): texts to convert.

        Returns:
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.
        """

        length = []
        result = []
        decode_flag = True if type(text[0]) == bytes else False

        for item in text:

            if decode_flag:
                item = item.decode('utf-8', 'strict')
            length.append(len(item))
            for char in item:
                index = self.dict[char]
                result.append(index)
        text = result
        return (torch.IntTensor(text), torch.IntTensor(length))

    def decode(self, t, length, raw=False):
        """Decode encoded texts back into strs.

        Args:
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.

        Raises:
            AssertionError: when the texts and its length does not match.

        Returns:
            text (str or list of str): texts to convert.
        """
        if length.numel() == 1:
            length = length[0]
            assert t.numel() == length, "text with length: {} does not match declared length: {}".format(t.numel(),
                                                                                                         length)
            if raw:
                return ''.join([self.alphabet[i - 1] for i in t])
            else:
                char_list = []
                for i in range(length):
                    if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):
                        char_list.append(self.alphabet[t[i] - 1])
                return ''.join(char_list)
        else:
            # batch mode
            assert t.numel() == length.sum(), "texts with length: {} does not match declared length: {}".format(
                t.numel(), length.sum())
            texts = []
            index = 0
            for i in range(length.numel()):
                l = length[i]
                texts.append(
                    self.decode(
                        t[index:index + l], torch.IntTensor([l]), raw=raw))
                index += l
            return texts

class Plot():
    def __init__(self,nepoch,fname='data/loss.png'):
        self.loss=[]
        self.accs=[]
        self.errs = []
        self.accs_index = []
        self.nepoch=nepoch
        self.fname=fname
        pass

    def add_loss(self,loss):
        self.loss.append(loss)

    def add_acc(self,acc,epoch):
        self.accs.append(acc)
        self.accs_index.append(epoch)

    def show(self):
        loss_x=[ (i+1)*(self.nepoch/len(self.loss)) for i in range(len(self.loss))]
        loss_y=self.loss

        accs_x=self.accs_index
        accs_y=self.accs

        errs_x = self.accs_index
        errs_y = self.errs

        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax1.plot(loss_x, loss_y, 'b-', label='loss')
        ax2.plot(accs_x, accs_y, 'g-', label='accuracy')
        ax2.plot(errs_x, errs_y, 'r-', label='error rate')
        ax1.set_xlabel("epoch index")
        ax1.set_ylabel("loss", color='b')
        ax2.set_ylabel("accuracy/error")

        # plt.plot(loss_x,self.loss)
        plt.savefig(self.fname,dpi=500)
        plt.show()

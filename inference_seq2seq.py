import argparse
from PIL import Image

import torch

import utils_seq2seq as utils
import dataset_seq2seq as dataset

import models.seq2seq as crnn

parser = argparse.ArgumentParser()
parser.add_argument('--alphabet', default='data/images/alphabet.txt', help='')
parser.add_argument('--img_path', type=str, default='data/images/test/test_0.png', help='the path of the input image to network')
parser.add_argument('--imgH', type=int, default=32, help='the height of the input image to network')
parser.add_argument('--imgW', type=int, default=280, help='the width of the input image to network')
parser.add_argument('--hidden_size', type=int, default=256, help='size of the lstm hidden state')
parser.add_argument('--encoder', type=str, default='expr/encoder_best.pth', help="path to encoder (to continue training)")
parser.add_argument('--decoder', type=str, default='expr/decoder_best.pth', help='path to decoder (to continue training)')
parser.add_argument('--max_width', type=int, default=71, help='the width of the feature map out from cnn')
parser.add_argument('--mean_std_file', type=str, default='data/images/desc/mean_std.json', help='whether use gpu')
arg = parser.parse_args()


# load alphabet
with open(arg.alphabet,encoding='utf-8') as f:
    alphabet = f.read()

# define convert bwteen string and label index
converter = utils.ConvertBetweenStringAndLabel(alphabet)

# len(alphabet) + SOS_TOKEN + EOS_TOKEN
num_classes = len(alphabet) + 2

transformer = dataset.ResizeNormalize(img_width=arg.imgW, img_height=arg.imgH, mean_std_file=arg.mean_std_file)


def seq2seq_decode(encoder_out, decoder, decoder_input, decoder_hidden, max_length):
    decoded_words = []
    prob = 1.0
    for di in range(max_length):
        decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_out)
        probs = torch.exp(decoder_output)
        _, topi = decoder_output.data.topk(1)
        ni = topi.squeeze(1)
        decoder_input = ni
        prob *= probs[:, ni]
        if ni == utils.EOS_TOKEN:
            break
        else:
            decoded_words.append(converter.decode(ni))

    words = ''.join(decoded_words)
    prob = prob.item()

    return words, prob


def main():
    image = Image.open(arg.img_path).convert('RGB')
    image = transformer(image)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    image = image.to(device)
    image = image.view(1, *image.size())
    image = torch.autograd.Variable(image)

    encoder = crnn.Encoder(3, arg.hidden_size).to(device)
    # no dropout during inference
    decoder = crnn.Decoder(arg.hidden_size, num_classes, dropout_p=0.0, max_length=arg.max_width).to(device)


    encoder.load_state_dict(torch.load(arg.encoder, map_location=device))
    print('loading pretrained encoder models from {}.'.format(arg.encoder))
    decoder.load_state_dict(torch.load(arg.decoder, map_location=device))
    print('loading pretrained decoder models from {}.'.format(arg.decoder))

    encoder.eval()
    decoder.eval()

    encoder_out = encoder(image)

    max_length = 20
    decoder_input = torch.zeros(1).long().to(device)
    decoder_hidden = decoder.initHidden(1).to(device)

    words, prob = seq2seq_decode(encoder_out, decoder, decoder_input, decoder_hidden, max_length)
    print('predict_string: {} => predict_probility: {}'.format(words, prob))


if __name__ == "__main__":
    main()
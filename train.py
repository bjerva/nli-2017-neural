from __future__ import print_function
import argparse
import os
import random
random.seed(1000)
import numpy as np
np.random.seed(1000)

import chainer
import chainer.links as L
import chainer.functions as F
from chainer import training
from chainer.training import extensions

from datasets import IMDBDataset, NLIDataset

from collections import defaultdict

class RNNModel(chainer.Chain):
    def __init__(self, vocab_size, n_labels, out_size, hidden_size, dropout):
        super().__init__(
            embed=L.EmbedID(vocab_size, out_size, ignore_label=-1),
            char_lstm=L.LSTM(out_size, hidden_size//2),
            word_lstm=L.LSTM(hidden_size//2, hidden_size),
            fc=L.Linear(None, n_labels)
        )
        self.dropout = dropout
        self.train = True

    def reset_state(self):
        self.char_lstm.reset_state()
        self.word_lstm.reset_state()

    def __call__(self, x):
        # hs = []
        # for sent in x:
        #     self.word_lstm.reset_state()
        #     word_reps = []
        #     for word in sent:
        #         self.char_lstm.reset_state()
        #         h = F.dropout(self.embed(word), self.dropout, self.train)
        #         h = F.dropout(self.char_lstm(h), self.dropout, self.train)
        #         word_reps.append(h)
        #
        #     sent_rep = F.vstack(word_reps)
        #     h = F.dropout(self.word_lstm(sent_rep), self.dropout, self.train)
        #     hs.append(h[-1])
        #
        # return self.fc(F.vstack(hs))
        h = self.embed(x)
        #import pdb; pdb.set_trace()
        self.char_lstm.reset_state()
        for idx, seq in enumerate(F.transpose_sequence(h)):
            h = self.char_lstm(seq)

        #print('b')

        return self.fc(h)

class CNNModel(chainer.Chain):
    def __init__(self, vocab_size, n_labels, out_size, hidden_size, dropout):
        output_channel = hidden_size
        sent_len = args.maxlen
        out_channels = int(args.maxlen*2)
        super().__init__(
            embed=L.EmbedID(vocab_size, out_size, ignore_label=-1),
            # conv1 = L.Convolution2D(sent_len, out_channels, (2, 2)),
            # conv2 = L.Convolution2D(out_channels, out_channels*2, (2, 2)),
            # l1    = L.Linear(480, hidden_size),
            # fc=L.Linear(None, n_labels)

            # Block 1
            bn0 = L.BatchNormalization(out_size),
            conv1 = L.ConvolutionND(ndim=1,
                in_channels=out_size, out_channels=out_size, ksize=2, stride=2, cover_all=True),
            bn1 = L.BatchNormalization(out_size),
            conv2 = L.ConvolutionND(ndim=1,
                in_channels=out_size, out_channels=out_size, ksize=2, stride=2, cover_all=True),
            bn2 = L.BatchNormalization(out_size),

            # Block 2
            bn3 = L.BatchNormalization(out_size*2),
            conv3 = L.ConvolutionND(ndim=1,
                in_channels=out_size*2, out_channels=out_size*2, ksize=2, stride=2, cover_all=True),
            bn4 = L.BatchNormalization(out_size*2),
            conv4 = L.ConvolutionND(ndim=1,
                in_channels=out_size*2, out_channels=out_size*2, ksize=2, stride=2, cover_all=True),

            # Block 3
            bn5 = L.BatchNormalization(out_size*4),
            conv5 = L.ConvolutionND(ndim=1,
                in_channels=out_size*4, out_channels=out_size*4, ksize=2, stride=2, cover_all=True),
            bn6 = L.BatchNormalization(out_size*4),
            conv6 = L.ConvolutionND(ndim=1,
                in_channels=out_size*4, out_channels=out_size*4, ksize=2, stride=2, cover_all=True),

            # Block 4
            bn7 = L.BatchNormalization(out_size*8),
            conv7 = L.ConvolutionND(ndim=1,
                in_channels=out_size*8, out_channels=out_size*8, ksize=2, stride=2, cover_all=True),
            bn8 = L.BatchNormalization(out_size*8),
            conv8 = L.ConvolutionND(ndim=1,
                in_channels=out_size*8, out_channels=out_size*8, ksize=2, stride=2, cover_all=True),

            # Block 5
            bn9 = L.BatchNormalization(out_size*16),
            conv9 = L.ConvolutionND(ndim=1,
                in_channels=out_size*16, out_channels=out_size*16, ksize=2, stride=2, cover_all=True),
            bn10 = L.BatchNormalization(out_size*16),
            conv10 = L.ConvolutionND(ndim=1,
                in_channels=out_size*16, out_channels=out_size*16, ksize=2, stride=2, cover_all=True),

            # Fully connected
            #fc4 = L.Linear(None, 1024),
            fc5 = L.Linear(None, 512),
            fc6 = L.Linear(512, 128),
            fc7 = L.Linear(128, n_labels)
        )
        self.dropout = dropout
        self.train = True
        self.first = True

    def __call__(self, x):
        #self.rnn1.reset_state()

        h = self.embed(x)
        if self.first:
            print('emb', h.data.shape)

        h = F.swapaxes(h, 1, 2)
        h = F.dropout(h, self.dropout)
        prev_h = h

        #### Block 1 ####
        if args.bn:
            h = self.bn0(h)
        if args.activation:
            h = F.relu(h)
        #h = F.dropout(h, self.dropout)
        if self.first:
            print('swp', h.data.shape)
        h = self.conv1(h)
        if self.first:
            print('cv1', h.data.shape)

        h = F.dropout(h, self.dropout)

        if args.bn:
            h = self.bn1(h)
        if args.activation:
            h = F.relu(h)
        h = self.conv2(h)
        if self.first:
            print('cv2', h.data.shape)

        h = F.average_pooling_nd(h, 2)
        if self.first:
            print('av1', h.data.shape)


        prev_h = F.average_pooling_nd(prev_h, 8)
        h = F.concat((h, prev_h))
        prev_h = h
        if self.first:
            print('rn1', h.data.shape)

        #### Block 2 ####
        if args.bn:
            h = self.bn3(h)
        if args.activation:
            h = F.relu(h)
        h = self.conv3(h)
        if self.first:
            print('cv3', h.data.shape)

        h = F.dropout(h, self.dropout)

        if args.bn:
            h = self.bn4(h)
        if args.activation:
            h = F.relu(h)
        h = self.conv4(h)
        if self.first:
            print('cv4', h.data.shape)


        prev_h = F.average_pooling_nd(prev_h, 4)
        if self.first:
            print('av2', prev_h.data.shape)

        h = F.concat((h, prev_h))
        prev_h = h
        if self.first:
            print('rn2', h.data.shape)


        #### Block 3 ####
        if args.bn:
            h = self.bn5(h)
        if args.activation:
            h = F.relu(h)
        h = self.conv5(h)
        if self.first:
            print('cv5', h.data.shape)

        h = F.dropout(h, self.dropout)

        if args.bn:
            h = self.bn6(h)
        if args.activation:
            h = F.relu(h)
        h = self.conv6(h)
        if self.first:
            print('cv6', h.data.shape)


        prev_h = F.average_pooling_nd(prev_h, 4)
        if self.first:
            print('av3', prev_h.data.shape)

        h = F.concat((h, prev_h))
        prev_h = h
        if self.first:
            print('mr3', h.data.shape)


        #### Block 4 ####
        if args.bn:
            h = self.bn7(h)
        if args.activation:
            h = F.relu(h)
        h = self.conv7(h)
        if self.first:
            print('cv7', h.data.shape)

        h = F.dropout(h, self.dropout)

        if args.bn:
            h = self.bn8(h)
        if args.activation:
            h = F.relu(h)
        h = self.conv8(h)
        if self.first:
            print('cv8', h.data.shape)


        prev_h = F.average_pooling_nd(prev_h, 4)
        if self.first:
            print('av4', prev_h.data.shape)

        h = F.concat((h, prev_h))
        prev_h = h
        if self.first:
            print('mr4', h.data.shape)


        # #### Block 5 ####
        # if args.bn:
        #     h = self.bn9(h)
        # if args.activation:
        #     h = F.relu(h)
        # h = self.conv9(h)
        # if self.first:
        #     print('cv9', h.data.shape)
        #
        #
        # h = F.dropout(h, self.dropout)
        #
        # if args.bn:
        #     h = self.bn10(h)
        # if args.activation:
        #     h = F.relu(h)
        # h = self.conv10(h)
        # if self.first:
        #     print('cv10', h.data.shape)
        #
        #
        # prev_h = F.average_pooling_nd(prev_h, 4)
        # if self.first:
        #     print('av4', prev_h.data.shape)
        #
        # h = F.concat((h, prev_h))
        # prev_h = h
        # if self.first:
        #     print('mr4', h.data.shape)


        #### Fully Connected ####
        if self.first:
            print('Hidden has to deal with {0} units as input'.format(h.shape[1]*h.shape[2]))
        h = F.dropout(h, self.dropout)
        h = F.relu(h)
        h = self.fc5(h)
        if self.first:
            print('fc5', h.data.shape)

        h = F.dropout(h, self.dropout)
        h = F.relu(h)
        h = self.fc6(h)
        if self.first:
           print('fc6', h.data.shape)

        h = F.relu(h)
        h = F.dropout(h, self.dropout)
        h = self.fc7(h)
        if self.first:
            self.first = False
        return h


class FFNNModel(chainer.Chain):
    def __init__(self, vocab_size, n_labels, hidden_size, dropout):
        super().__init__(
            fc0 = L.Linear(None, 2048),
            fc1 = L.Linear(None, 512),
            fc2 = L.Linear(512, 128),
            fc3 = L.Linear(128, n_labels)
        )
        self.dropout = dropout
        self.train = True
        self.first = True

    def __call__(self, x):
        h = self.fc0(x)
        h = self.fc1(F.dropout(h, self.dropout, self.train))
        h = self.fc2(F.dropout(h, self.dropout, self.train))
        #h = self.fc2(F.dropout(h, self.dropout, self.train))
        h = self.fc3(F.dropout(h, self.dropout, self.train))
        return h
        # h = F.relu(h)
        # h = self.fc5(F.dropout(h, self.dropout, self.train))
        # if self.first:
        #     print('fc5', h.data.shape)
        #
        # h = F.relu(h)
        # h = self.fc6(F.dropout(h, self.dropout, self.train))
        # if self.first:
        #    print('fc6', h.data.shape)
        #
        # h = F.relu(h)
        # if self.first:
        #     #import pdb; pdb.set_trace()
        #     self.first = False
        # return self.fc7(F.dropout(h, self.dropout, self.train))
# epoch       main/loss   validation/main/loss  main/accuracy  validation/main/accuracy
# 1           1.26909     0.827249              0.722          0.683333
# 2           0.628819    0.570169              0.835          0.82
# 3           0.304336    0.545898              0.903667       0.866667
# 4           0.382191    1.22191               0.902          0.726667
# 5           0.205926    0.700033              0.935667       0.843333

class TestModeEvaluator(extensions.Evaluator):

    def evaluate(self):
        model = self.get_target('main')
        model.train = False
        ret = super().evaluate()
        model.train = True
        return ret


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchsize', '-b', type=int, default=24,
                        help='Number of documents in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=20,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--embeddings', default='',
                        help='Iinitial word embeddings file')
    parser.add_argument('--vocabulary', default='',
                        help='Vocabulary file')
    parser.add_argument('--dataset', default='data/aclImdb', type=str,
                        help='IMDB dataset path, dir with train/ and test/ folders')
    parser.add_argument('--vocab_size', default=68379, type=int,
                        help='GloVe word embedding dimensions')
    parser.add_argument('--out_size', default=300, type=int,
                        help='GloVe word embedding dimensions')
    parser.add_argument('--hidden_size', default=256, type=int,
                        help='Hidden layers dimensions')
    parser.add_argument('--maxlen', default=400, type=int,
                        help='Maximum sequence time (T) length')
    parser.add_argument('--dropout', default=0.3, type=float,
                        help='Dropout ratio between layers')
    parser.add_argument('--activation', action='store_true')
    parser.add_argument('--bn', action='store_true')
    return parser.parse_args()


args = parse_args()
def main():


    char_to_id = defaultdict(lambda: len(char_to_id))
    label_to_id = defaultdict(lambda: len(label_to_id))
    unk = char_to_id['<UNK>']
    bos = char_to_id['<S>']
    eos = char_to_id['</S>']
    if 'Imdb' in args.dataset:
        train, test = IMDBDataset(os.path.join(args.dataset, 'train'), args.vocabulary, char_to_id, args.maxlen),\
                      IMDBDataset(os.path.join(args.dataset, 'test'), args.vocabulary, char_to_id, args.maxlen)

    elif 'nli' in args.dataset:
        print('nli!')
        train = NLIDataset(args.dataset, 'train', char_to_id, label_to_id, args.maxlen, args.batchsize, repeat=True)
        test = NLIDataset(args.dataset, 'dev', char_to_id, label_to_id, args.maxlen, args.batchsize, repeat=False, shuffle=False)

    vocab_size = len(char_to_id) + 1#max(map(max, train.pos_dataset))
    n_labels = 2 if len(label_to_id) == 0 else len(label_to_id)

    print('{0} chars'.format(vocab_size))
    print('{0} labels'.format(n_labels))
    # model = L.Classifier(QRNNModel(
    #    vocab_size, n_labels, args.out_size, args.hidden_size, args.dropout))

    # model = L.Classifier(RNNModel(
    #    vocab_size, n_labels, args.out_size, args.hidden_size, args.dropout))

    model = L.Classifier(CNNModel(
        vocab_size, n_labels, args.out_size, args.hidden_size, args.dropout))

    # model = L.Classifier(FFNNModel(
    #     vocab_size, n_labels,args.hidden_size, args.dropout))

    if args.embeddings:
        model.predictor.embed.W.data = util.load_embeddings(
            args.embeddings, args.vocab_size, args.out_size)

    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        model.to_gpu()
        model.predictor.embed.to_gpu()
        # model.predictor.char_lstm.to_gpu()

    optimizer = chainer.optimizers.Adam()#RMSprop(lr=0.001, alpha=0.9)
    optimizer.use_cleargrads()
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(1e-4))

    if 'Imdb' in args.dataset:
        train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
        test_iter = chainer.iterators.SerialIterator(test, 1,
                                                     repeat=False, shuffle=False)
    elif 'nli' in args.dataset:
        train_iter = train
        test_iter = test

    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    trainer.extend(TestModeEvaluator(test_iter, model, device=args.gpu))
    #trainer.extend(extensions.ExponentialShift('lr', 0.5), trigger=(25, 'epoch'))
    trainer.extend(extensions.snapshot(), trigger=(args.epoch, 'epoch'))
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss',
         'main/accuracy', 'validation/main/accuracy']))
    trainer.extend(extensions.ProgressBar(update_interval=10))

    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)

    trainer.run()

if __name__ == '__main__':
    main()

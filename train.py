from __future__ import print_function
import argparse
import os
import random
# random.seed(1000)
import numpy as np
# np.random.seed(1000)

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
    def __init__(self, vocab_size, n_labels, word_vocab_size, n_pos, out_size, dropout, args):
        self.args = args
        sent_len = 4096#args.maxlen
        out_channels = int(sent_len*2)
        super().__init__(
            embed_tri=L.EmbedID(vocab_size[0], out_size, ignore_label=-1),
            embed_four=L.EmbedID(vocab_size[1], out_size, ignore_label=-1),
            embed_word=L.EmbedID(word_vocab_size, out_size, ignore_label=-1),
            embed_pos=L.EmbedID(n_pos, out_size, ignore_label=-1),

            # Block 1
            bn1 = L.BatchNormalization(out_size),
            conv1 = L.ConvolutionND(ndim=1,
                in_channels=out_size, out_channels=out_size, ksize=3, stride=2, cover_all=True),
            bn2 = L.BatchNormalization(out_size),
            conv2 = L.ConvolutionND(ndim=1,
                in_channels=out_size, out_channels=out_size, ksize=2, stride=2, cover_all=True),

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

            # Block 1
            bn1_b = L.BatchNormalization(out_size),
            conv1_b = L.ConvolutionND(ndim=1,
                in_channels=out_size, out_channels=out_size, ksize=3, stride=2, cover_all=True),
            bn2_b = L.BatchNormalization(out_size),
            conv2_b = L.ConvolutionND(ndim=1,
                in_channels=out_size, out_channels=out_size, ksize=2, stride=2, cover_all=True),

            # Block 2
            bn3_b = L.BatchNormalization(out_size*2),
            conv3_b = L.ConvolutionND(ndim=1,
                in_channels=out_size*2, out_channels=out_size*2, ksize=2, stride=2, cover_all=True),
            bn4_b = L.BatchNormalization(out_size*2),
            conv4_b = L.ConvolutionND(ndim=1,
                in_channels=out_size*2, out_channels=out_size*2, ksize=2, stride=2, cover_all=True),

            # Block 3
            bn5_b = L.BatchNormalization(out_size*4),
            conv5_b = L.ConvolutionND(ndim=1,
                in_channels=out_size*4, out_channels=out_size*4, ksize=2, stride=2, cover_all=True),
            bn6_b = L.BatchNormalization(out_size*4),
            conv6_b = L.ConvolutionND(ndim=1,
                in_channels=out_size*4, out_channels=out_size*4, ksize=2, stride=2, cover_all=True),

            # Block 4
            bn7_b = L.BatchNormalization(out_size*8),
            conv7_b = L.ConvolutionND(ndim=1,
                in_channels=out_size*8, out_channels=out_size*8, ksize=2, stride=2, cover_all=True),
            bn8_b = L.BatchNormalization(out_size*8),
            conv8_b = L.ConvolutionND(ndim=1,
                in_channels=out_size*8, out_channels=out_size*8, ksize=2, stride=2, cover_all=True),

            # Block 1
            bn1_w = L.BatchNormalization(out_size),
            conv1_w = L.ConvolutionND(ndim=1,
                in_channels=out_size, out_channels=out_size, ksize=2, stride=2, cover_all=True),
            bn2_w = L.BatchNormalization(out_size),
            conv2_w = L.ConvolutionND(ndim=1,
                in_channels=out_size, out_channels=out_size, ksize=2, stride=2, cover_all=True),

            # Block 2
            bn3_w = L.BatchNormalization(out_size*2),
            conv3_w = L.ConvolutionND(ndim=1,
                in_channels=out_size*2, out_channels=out_size*2, ksize=2, stride=2, cover_all=True),
            bn4_w = L.BatchNormalization(out_size*2),
            conv4_w = L.ConvolutionND(ndim=1,
                in_channels=out_size*2, out_channels=out_size*2, ksize=2, stride=2, cover_all=True),

            # Block 1
            bn1_p = L.BatchNormalization(out_size),
            conv1_p = L.ConvolutionND(ndim=1,
                in_channels=out_size, out_channels=out_size, ksize=2, stride=2, cover_all=True),
            bn2_p = L.BatchNormalization(out_size),
            conv2_p = L.ConvolutionND(ndim=1,
                in_channels=out_size, out_channels=out_size, ksize=2, stride=2, cover_all=True),

            # Block 2
            bn3_p = L.BatchNormalization(out_size*2),
            conv3_p = L.ConvolutionND(ndim=1,
                in_channels=out_size*2, out_channels=out_size*2, ksize=2, stride=2, cover_all=True),
            bn4_p = L.BatchNormalization(out_size*2),
            conv4_p = L.ConvolutionND(ndim=1,
                in_channels=out_size*2, out_channels=out_size*2, ksize=2, stride=2, cover_all=True),

            fcb1 = L.Linear(None, 1024),
            fcb2 = L.Linear(1024, 128),
            # Fully connected
            #fc3 = L.Linear(None, 2048),
            #fctri  = L.Linear(None, 2048),
            fcfour = L.Linear(None, 2048),
            fcword = L.Linear(None, 1024),
            #fcpos = L.Linear(None, 1024),
            #fc4 = L.Linear(None, 1024),
            fc5 = L.Linear(None, 1024),
            fc6 = L.Linear(None, 256),
            fc7 = L.Linear(None, n_labels),
        )
        self.dropout = dropout
        self.train = True
        self.first = True

    def call_bow(self, x):
        h = F.dropout(x, self.dropout)
        h = self.fcb1(h)
        if self.first:
            print('\tfcb1', h.data.shape)
        h = F.dropout(h, 0.8)
        h = F.relu(h)
        h = self.fcb2(h)
        if self.first:
            print('\tfcb2', h.data.shape)
        h = F.dropout(h, self.dropout)
        h = F.relu(h)

        return h

    def call_trinet(self, h):
        prev_h = h

        #### Block 1 ####
        if self.args.bn:
            h = self.bn1(h)
        if self.args.activation:
            h = F.relu(h)
        #h = F.dropout(h, self.dropout)
        if self.first:
            print('### 3-net ###')
            print('inp', h.data.shape)
        h = self.conv1(h)
        if self.first:
            print('cv1', h.data.shape)

        h = F.dropout(h, self.dropout)

        if self.args.bn:
            h = self.bn2(h)
        if self.args.activation:
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
        if self.args.bn:
            h = self.bn3(h)
        if self.args.activation:
            h = F.relu(h)
        h = self.conv3(h)
        if self.first:
            print('cv3', h.data.shape)

        h = F.dropout(h, self.dropout)

        if self.args.bn:
            h = self.bn4(h)
        if self.args.activation:
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
        if self.args.bn:
            h = self.bn5(h)
        if self.args.activation:
            h = F.relu(h)
        h = self.conv5(h)
        if self.first:
            print('cv5', h.data.shape)

        h = F.dropout(h, self.dropout)

        if self.args.bn:
            h = self.bn6(h)
        if self.args.activation:
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
        if self.args.bn:
            h = self.bn7(h)
        if self.args.activation:
            h = F.relu(h)
        h = self.conv7(h)
        if self.first:
            print('cv7', h.data.shape)

        h = F.dropout(h, self.dropout)

        if self.args.bn:
            h = self.bn8(h)
        if self.args.activation:
            h = F.relu(h)
        h = self.conv8(h)
        if self.first:
            print('cv8', h.data.shape)


        prev_h = F.average_pooling_nd(prev_h, 4)
        if self.first:
            print('av4', prev_h.data.shape)

        h = F.concat((h, prev_h))
        if self.first:
            print('mr4', h.data.shape)

        h = F.dropout(h, self.dropout)
        h = F.relu(h)
        h = self.fctri(h)
        if self.first:
            print('fctri', h.data.shape)

        return h

    def call_fournet(self, h):
        prev_h = h

        #### Block 1 ####
        if self.args.bn:
            h = self.bn1_b(h)
        if self.args.activation:
            h = F.relu(h)
        #h = F.dropout(h, self.dropout)
        if self.first:
            print('### 4-net ###')
            print('inp', h.data.shape)
        h = self.conv1_b(h)
        if self.first:
            print('cv1', h.data.shape)

        h = F.dropout(h, self.dropout)

        if self.args.bn:
            h = self.bn2_b(h)
        if self.args.activation:
            h = F.relu(h)
        h = self.conv2_b(h)
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
        if self.args.bn:
            h = self.bn3_b(h)
        if self.args.activation:
            h = F.relu(h)
        h = self.conv3_b(h)
        if self.first:
            print('cv3', h.data.shape)

        h = F.dropout(h, self.dropout)

        if self.args.bn:
            h = self.bn4_b(h)
        if self.args.activation:
            h = F.relu(h)
        h = self.conv4_b(h)
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
        if self.args.bn:
            h = self.bn5_b(h)
        if self.args.activation:
            h = F.relu(h)
        h = self.conv5_b(h)
        if self.first:
            print('cv5', h.data.shape)

        h = F.dropout(h, self.dropout)

        if self.args.bn:
            h = self.bn6_b(h)
        if self.args.activation:
            h = F.relu(h)
        h = self.conv6_b(h)
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
        if self.args.bn:
            h = self.bn7_b(h)
        if self.args.activation:
            h = F.relu(h)
        h = self.conv7_b(h)
        if self.first:
            print('cv7', h.data.shape)

        h = F.dropout(h, self.dropout)

        if self.args.bn:
            h = self.bn8_b(h)
        if self.args.activation:
            h = F.relu(h)
        h = self.conv8_b(h)
        if self.first:
            print('cv8', h.data.shape)

        prev_h = F.average_pooling_nd(prev_h, 4)
        if self.first:
            print('av4', prev_h.data.shape)

        h = F.concat((h, prev_h))
        if self.first:
            print('mr4', h.data.shape)

        h = F.dropout(h, self.dropout)
        h = F.relu(h)
        h = self.fcfour(h)
        if self.first:
            print('fcfour', h.data.shape)

        return h

    def call_wordnet(self, h):
        prev_h = h

        #### Block 1 ####
        if self.args.bn:
            h = self.bn1_w(h)
        if self.args.activation:
            h = F.relu(h)
        #h = F.dropout(h, self.dropout)
        if self.first:
            print('### word-net ###')
            print('inp', h.data.shape)
        h = self.conv1_w(h)
        if self.first:
            print('cv1', h.data.shape)

        h = F.dropout(h, self.dropout+0.1)

        if self.args.bn:
            h = self.bn2_w(h)
        if self.args.activation:
            h = F.relu(h)
        h = self.conv2_w(h)
        if self.first:
            print('cv2', h.data.shape)

        h = F.average_pooling_nd(h, 2)
        if self.first:
            print('av1', h.data.shape)

        h = F.dropout(h, self.dropout+0.1)

        prev_h = F.average_pooling_nd(prev_h, 8)
        h = F.concat((h, prev_h))
        if self.first:
            print('rn1', h.data.shape)

        prev_h = h

        #### Block 2 ####
        if self.args.bn:
            h = self.bn3_w(h)
        if self.args.activation:
            h = F.relu(h)
        h = self.conv3_w(h)
        if self.first:
            print('cv3', h.data.shape)

        h = F.dropout(h, self.dropout)

        if self.args.bn:
            h = self.bn4_w(h)
        if self.args.activation:
            h = F.relu(h)
        h = self.conv4_w(h)
        if self.first:
            print('cv4', h.data.shape)

        h = F.dropout(h, self.dropout)

        prev_h = F.average_pooling_nd(prev_h, 4)
        if self.first:
            print('av2', prev_h.data.shape)

        h = F.concat((h, prev_h))
        prev_h = h
        if self.first:
            print('rn2', h.data.shape)

        h = F.dropout(h, self.dropout)
        h = F.relu(h)
        h = self.fcword(h)
        if self.first:
            print('fcword', h.data.shape)

        return h


    def call_posnet(self, h):
        prev_h = h

        #### Block 1 ####
        if self.args.bn:
            h = self.bn1_p(h)
        if self.args.activation:
            h = F.relu(h)
        #h = F.dropout(h, self.dropout)
        if self.first:
            print('### word-net ###')
            print('inp', h.data.shape)
        h = self.conv1_p(h)
        if self.first:
            print('cv1', h.data.shape)

        h = F.dropout(h, self.dropout+0.1)

        if self.args.bn:
            h = self.bn2_p(h)
        if self.args.activation:
            h = F.relu(h)
        h = self.conv2_p(h)
        if self.first:
            print('cv2', h.data.shape)

        h = F.average_pooling_nd(h, 2)
        if self.first:
            print('av1', h.data.shape)

        h = F.dropout(h, self.dropout+0.1)

        prev_h = F.average_pooling_nd(prev_h, 8)
        h = F.concat((h, prev_h))
        if self.first:
            print('rn1', h.data.shape)

        prev_h = h

        #### Block 2 ####
        if self.args.bn:
            h = self.bn3_p(h)
        if self.args.activation:
            h = F.relu(h)
        h = self.conv3_p(h)
        if self.first:
            print('cv3', h.data.shape)

        h = F.dropout(h, self.dropout)

        if self.args.bn:
            h = self.bn4_p(h)
        if self.args.activation:
            h = F.relu(h)
        h = self.conv4_p(h)
        if self.first:
            print('cv4', h.data.shape)

        h = F.dropout(h, self.dropout)

        prev_h = F.average_pooling_nd(prev_h, 4)
        if self.first:
            print('av2', prev_h.data.shape)

        h = F.concat((h, prev_h))
        prev_h = h
        if self.first:
            print('rn2', h.data.shape)

        h = F.dropout(h, self.dropout)
        h = F.relu(h)
        h = self.fcpos(h)
        if self.first:
            print('fcpos', h.data.shape)

        return h


    def __call__(self, x):
        if self.args.use_bow:
            x_bow = x[:,1024+8192:]
            x = x[:,:1024+8192]
            x = F.cast(x, np.int32)

        hs = []
        if self.args.use_tri:
            x_tri = x[:,:4096]
            h = self.embed_tri(x_tri)
            if self.first:
                print('emb', h.data.shape)

            h = F.swapaxes(h, 1, 2)
            h = F.dropout(h, self.dropout)
            h_tri = self.call_trinet(h)
            hs.append(h_tri)

        if self.args.use_four:
            x_four = x[:,4096:8192]
            h = self.embed_four(x_four)
            if self.first:
                print('emb', h.data.shape)

            h = F.swapaxes(h, 1, 2)
            h = F.dropout(h, self.dropout)
            h_four = self.call_fournet(h)
            hs.append(h_four)

        if self.args.use_words:
            x_words = x[:,8192:8192+1024]
            h = self.embed_word(x_words)
            h = F.swapaxes(h, 1, 2)
            h = F.dropout(h, self.dropout+0.1)
            h_words = self.call_wordnet(h)
            hs.append(h_words)

        # if self.args.use_pos:
        #     x_pos = x[:,1024+8192:8192+2048]
        #     h = self.embed_pos(x_pos)
        #     h = F.swapaxes(h, 1, 2)
        #     h = F.dropout(h, self.dropout+0.1)
        #     h_pos = self.call_posnet(h)
        #     hs.append(h_pos)

        if len(hs) > 1:
            h = F.concat(hs)
        else:
            h = hs[0]

        #h = F.average_pooling_nd(h, 2)

        #### Fully Connected ####
        if self.first:
            print('After concat: {0}'.format(h.data.shape))
        #
        # h = F.dropout(h, self.dropout)
        # h = F.relu(h)
        # h = self.fc4(h)
        # if self.first:
        #     print('fc4', h.data.shape)

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

        if self.args.use_bow:
            h_bows = self.call_bow(x_bow)
            h = F.concat((h, h_bows))
            if self.first:
                print('cnc', h.data.shape)

        h = F.relu(h)
        h = F.dropout(h, self.dropout)
        h = self.fc7(h)
        if self.first:
            print('out', h.data.shape)
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


class TestModeEvaluator(extensions.Evaluator):

    def evaluate(self):
        model = self.get_target('main')
        model.train = False
        ret = super().evaluate()
        model.train = True
        return ret

best_dev = 1000.0
def store_model(result):
    global best_dev
    dev_loss = result['validation/main/loss']
    if dev_loss < best_dev:
        best_dev = dev_loss
        chainer.serializers.save_npz('./models/' + args.name + '.best-loss.npz', model)
        with open('./logs/'+args.name+'_best_losses.txt', 'a') as out_f:
            out_f.write('{0}\n'.format(best_dev))



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
    parser.add_argument('--subset', type=int, default=0)
    parser.add_argument('--use_bow', action='store_true')
    parser.add_argument('--use_tri', action='store_true')
    parser.add_argument('--use_four', action='store_true')
    parser.add_argument('--use_words', action='store_true')
    parser.add_argument('--use_pos', action='store_true')
    parser.add_argument('--name', type=str)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    char_to_id = [defaultdict(lambda: len(char_to_id[0])), defaultdict(lambda: len(char_to_id[1]))]
    word_to_id = defaultdict(lambda: len(word_to_id))
    label_to_id = defaultdict(lambda: len(label_to_id))
    pos_to_id = defaultdict(lambda: len(pos_to_id))
    if 'Imdb' in args.dataset:
        train, test = IMDBDataset(os.path.join(args.dataset, 'train'), args.vocabulary, char_to_id, args.maxlen),\
                      IMDBDataset(os.path.join(args.dataset, 'test'), args.vocabulary, char_to_id, args.maxlen)

    elif 'nli' in args.dataset:
        print('nli!')
        train = NLIDataset(args.dataset, 'train', char_to_id, label_to_id, word_to_id, pos_to_id, args.maxlen, args.batchsize, repeat=True, subset=args.subset, use_bow=args.use_bow)
        test = NLIDataset(args.dataset, 'dev', char_to_id, label_to_id, word_to_id, pos_to_id, args.maxlen, args.batchsize, repeat=False, shuffle=False, subset=args.subset, use_bow=args.use_bow)

    vocab_size = [len(char_to_id[0]) + 1, len(char_to_id[1]) + 1]#max(map(max, train.pos_dataset))
    word_vocab_size = len(word_to_id) + 1
    n_labels = 2 if len(label_to_id) == 0 else len(label_to_id)
    n_pos = len(pos_to_id)

    print('{0}, {1} char ids'.format(vocab_size[0], vocab_size[1]))
    print('{0} word ids'.format(word_vocab_size))
    print('{0} labels'.format(n_labels))
    print('{0} pos'.format(n_pos))
    # model = L.Classifier(QRNNModel(
    #    vocab_size, n_labels, args.out_size, args.hidden_size, args.dropout))

    # model = L.Classifier(RNNModel(
    #    vocab_size, n_labels, args.out_size, args.hidden_size, args.dropout))

    model = L.Classifier(CNNModel(
        vocab_size, n_labels, word_vocab_size, n_pos, args.out_size, args.dropout))

    # model = L.Classifier(FFNNModel(
    #     vocab_size, n_labels,args.hidden_size, args.dropout))

    if args.embeddings:
        model.predictor.embed.W.data = util.load_embeddings(
            args.embeddings, args.vocab_size, args.out_size)

    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        model.to_gpu()
        model.predictor.embed_tri.to_gpu()
        model.predictor.embed_four.to_gpu()
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
    trainer.extend(extensions.LogReport(postprocess=store_model,
                                    trigger=(1, 'epoch')))
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss',
         'main/accuracy', 'validation/main/accuracy']))
    trainer.extend(extensions.ProgressBar(update_interval=8))

    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)

    trainer.run()

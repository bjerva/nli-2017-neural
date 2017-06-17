import sys
import pickle
import os.path
import random
import argparse
from collections import namedtuple

import numpy as np

import chainer
from chainer import cuda, Variable
from chainer import optimizers, serializers
import chainer.functions as F

from dataset import NLIDataset
from model import MaxLSTMClassifier


Essay = namedtuple('Essay', ['label', 'sents'])


def main():
    parser = argparse.ArgumentParser(
            description='MaxLSTMClassifier training')
    parser.add_argument(
            '--data-path', type=str, metavar='FILE', required=True,
            help='path to nli-shared-task-2017 directory')
    parser.add_argument(
            '--embeddings', type=str, metavar='FILE', required=True,
            help='path to GloVe file (uncompressed)')
    parser.add_argument(
            '--model', type=str, metavar='FILE', required=True,
            help='prefix to model path')
    parser.add_argument(
            '--embeddings-size', type=int, metavar='N', default=300,
            help='dimensionality of GloVe embeddings')
    parser.add_argument(
            '--batch-size', type=int, metavar='N', default=64,
            help='batch size (number of essays)')
    parser.add_argument(
            '--gpu', type=int, metavar='N', default=-1,
            help='gpu to use (default: use CPU)')
    parser.add_argument(
            '--limit-dev', type=int, metavar='N',
            help='limit the number of development essays used during training')
    parser.add_argument(
            '--pos-embeddings-size', type=int, metavar='N', default=64,
            help='dimensionality of POS embeddings')
    parser.add_argument(
            '--lstm-size', type=int, metavar='N', default=256,
            help='number of LSTM units (per direction)')
    args = parser.parse_args()


    dataset = NLIDataset(args.data_path)
    glove_file = args.embeddings
    glove_dims = args.embeddings_size
    batch_size = args.batch_size
    gpu = args.gpu
    pos_embedding_size = args.pos_embeddings_size
    state_size = args.lstm_size
    limit_dev = args.limit_dev

    with open(glove_file) as f:
        glove_vocab = {line.split(' ', 1)[0] for line in f}

    def get_data(part):
        return [
            Essay(
                label,
                list(dataset.get_sents(part, label.test_taker_id)))
            for label in dataset.labels[part].values()]


    # Full, cased vocabulary for train+dev: 79112
    # Of which 47884 are in the largest GloVe model

    def load_cached(part):
        if os.path.exists(part+'.pickle'):
            with open(part+'.pickle', 'rb') as f:
                return pickle.load(f)
        else:
            data = get_data(part)
            with open(part+'.pickle', 'wb') as f:
                pickle.dump(data, f, -1)
            return data

    # TODO: would prefer test vocabulary in here as well, for simplicity
    dev_essays = load_cached('dev')
    train_essays = load_cached('train')

    vocab = {token for essay in train_essays + dev_essays
                   for sent in essay.sents
                   for token,lemma,tag in sent}

    vocab = ['<UNK>'] + sorted(vocab & glove_vocab)
    vocab_index = {s:i for i,s in enumerate(vocab)}

    pos_vocab = sorted({
        tag for essay in train_essays + dev_essays
            for sent in essay.sents
            for token,lemma,tag in sent})
    pos_vocab_index = {s:i for i,s in enumerate(pos_vocab)}

    lang_vocab = sorted({
        essay.label.L1 for essay in train_essays + dev_essays})
    lang_vocab_index = {s:i for i,s in enumerate(lang_vocab)}

    glove = np.zeros((len(vocab), glove_dims), dtype=np.float32)
    glove_idxs = set()
    with open(glove_file) as f:
        for line in f:
            word, vector = line.split(' ', 1)
            idx = vocab_index.get(word)
            if idx is not None:
                glove_idxs.add(idx)
                glove[idx,:] = np.fromstring(vector, sep=' ')
    print('Loaded %d vectors from GloVe file' % len(glove_idxs))

    model = MaxLSTMClassifier(
            len(vocab), len(pos_vocab), glove_dims, pos_embedding_size,
            state_size, len(lang_vocab),
            embeddings=glove)
    if gpu >= 0: model.to_gpu(gpu)
    xp = model.xp

    def encode_essay(essay):
        unk = vocab_index['<UNK>']
        sents_token = [
                xp.array([vocab_index.get(token, unk)
                          for token,lemma,tag in sent],
                         dtype=xp.int32)
                for sent in essay.sents]
        sents_pos = [
                xp.array([pos_vocab_index[tag] for token,lemma,tag in sent],
                         dtype=xp.int32)
                for sent in essay.sents]
        return sents_token, sents_pos

    optimizer = chainer.optimizers.Adam()
    optimizer.use_cleargrads()
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.GradientClipping(5.0))

    n_batches = 0

    while True:
        batch_essays = random.sample(train_essays, batch_size)
        batch_token, batch_pos = list(zip(*map(encode_essay, batch_essays)))
        target = xp.array(
                [lang_vocab_index[essay.label.L1] for essay in batch_essays],
                dtype=xp.int32)
        pred = model(batch_token, batch_pos)
        loss = F.softmax_cross_entropy(pred, target)
        print('TRAIN', cuda.to_cpu(loss.data))
        model.cleargrads()
        loss.backward()
        optimizer.update()

        best_dev_loss = float('inf')

        if n_batches % 1000 == 0:
            dev_loss = 0.0
            dev_pred = []
            dev_target = []
            with chainer.using_config('train', False):
                n_dev_essays = limit_dev if limit_dev else len(dev_essays)
                for i in range(0, n_dev_essays, batch_size):
                    batch_essays = dev_essays[i:i+batch_size]
                    batch_token, batch_pos = list(zip(*map(
                        encode_essay, batch_essays)))
                    target = xp.array(
                            [lang_vocab_index[essay.label.L1]
                             for essay in batch_essays],
                            dtype=xp.int32)
                    pred = model(batch_token, batch_pos)
                    loss = F.softmax_cross_entropy(pred, target)
                    dev_loss += cuda.to_cpu(loss.data)
                    dev_target.extend(cuda.to_cpu(target).tolist())
                    dev_pred.extend(
                            np.argmax(cuda.to_cpu(pred.data), axis=-1).tolist())

            print('DEV', dev_loss, n_batches)
            if dev_loss < best_dev_loss:
                best_dev_loss = dev_loss
                serializers.save_npz(args.model + '.npz', model)
                with open(args.model + '.metadata', 'wb') as f:
                    pickle.dump((args, vocab, pos_vocab, lang_vocab), f, -1)
                print('SAVE', n_batches)

            confusion = np.zeros((len(lang_vocab),)*2, dtype=np.int32)
            for x,y in zip(dev_pred, dev_target):
                confusion[x,y] += 1
            print('ACC',
                    sum(x==y for x,y in zip(dev_pred, dev_target))/
                    len(dev_target),
                    n_batches)
            print(confusion)
        n_batches += 1


if __name__ == '__main__': main()


import sys
import pickle
import os.path
import random
import argparse
from collections import namedtuple
import time

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
            '--test-path', type=str, metavar='FILE',
            help='path to nli-test-set_phase1_2017 directory')
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
    parser.add_argument(
            '--dropout-sentences', type=float, metavar='X', default=0.5)
    parser.add_argument(
            '--dropout-tokens', type=float, metavar='X', default=0.3)
    parser.add_argument(
            '--dropout', type=float, metavar='X', default=0.2)
    args = parser.parse_args()


    trainset = NLIDataset(args.data_path, ['dev', 'train'])
    if args.test_path:
        testset = NLIDataset(args.test_path, ['test'])
    else:
        testset = None
    glove_file = args.embeddings
    glove_dims = args.embeddings_size
    batch_size = args.batch_size
    gpu = args.gpu
    pos_embedding_size = args.pos_embeddings_size
    state_size = args.lstm_size
    limit_dev = args.limit_dev
    dropout_sentences = args.dropout_sentences
    dropout_tokens = args.dropout_tokens

    with open(glove_file) as f:
        glove_vocab = {line.split(' ', 1)[0] for line in f}

    def get_data(part):
        dataset = testset if part == 'test' else trainset
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

    dev_essays = load_cached('dev')
    train_essays = load_cached('train')
    if testset:
        test_essays = load_cached('test')
    else:
        test_essays = []

    vocab = {token for essay in train_essays + dev_essays + test_essays
                   for sent in essay.sents
                   for token,lemma,tag in sent}

    vocab = ['<UNK>'] + sorted(vocab & glove_vocab)
    vocab_index = {s:i for i,s in enumerate(vocab)}

    pos_vocab = sorted({
        tag for essay in train_essays + dev_essays + test_essays
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
    print('Loaded %d vectors from GloVe file' % len(glove_idxs), flush=True)

    model = MaxLSTMClassifier(
            len(vocab), len(pos_vocab), glove_dims, pos_embedding_size,
            state_size, len(lang_vocab),
            embeddings=glove, dropout=args.dropout)

    # Freeze vocabulary embeddings
    model.embeddings.disable_update()

    if gpu >= 0: model.to_gpu(gpu)
    xp = model.xp

    def encode_essay(essay, dropout_sentences=0, dropout_tokens=0):
        sents = essay.sents
        if dropout_sentences:
            n_sents = round(dropout_sentences * len(sents))
            if n_sents >= 1:
                sents = list(sents)
                random.shuffle(sents)
                sents = sents[:n_sents]
        unk = vocab_index['<UNK>']
        sents_token = [
                xp.array([vocab_index.get(token, unk)
                          for token,lemma,tag in sent],
                         dtype=xp.int32)
                for sent in sents]
        sents_pos = [
                xp.array([pos_vocab_index[tag] for token,lemma,tag in sent],
                         dtype=xp.int32)
                for sent in sents]
        if dropout_tokens:
            def remove_symbols(x):
                mask = np.random.random(len(x)) < dropout_tokens
                mask = xp.array(mask.astype(np.int32))
                return x*(1-mask) - mask
            sents_token = [remove_symbols(x) for x in sents_token]
            senst_pos = [remove_symbols(x) for x in sents_pos]
        return sents_token, sents_pos

    optimizer = chainer.optimizers.Adam()
    optimizer.use_cleargrads()
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.GradientClipping(5.0))

    n_batches = 0
    best_dev_loss = float('inf')

    while True:
        t0 = time.time()
        batch_essays = random.sample(train_essays, batch_size)
        batch_pairs = [encode_essay(essay, dropout_sentences=dropout_sentences)
                       for essay in batch_essays]
        batch_token, batch_pos = list(zip(*batch_pairs))
        target = xp.array(
                [lang_vocab_index[essay.label.L1] for essay in batch_essays],
                dtype=xp.int32)
        pred = model(batch_token, batch_pos)
        loss = F.softmax_cross_entropy(pred, target)
        model.cleargrads()
        loss.backward()
        optimizer.update()
        print('TRAIN', cuda.to_cpu(loss.data), time.time()-t0, flush=True)

        if n_batches % 200 == 0:
            t0 = time.time()
            dev_loss = 0.0
            dev_pred = []
            dev_target = []
            dev_result = {}
            with chainer.using_config('train', False):
                with chainer.using_config('enable_backprop', False):
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
                                np.argmax(cuda.to_cpu(pred.data),
                                    axis=-1).tolist())
                        pred = F.softmax(pred)
                        pred = cuda.to_cpu(pred.data)
                        for essay, ps in zip(batch_essays, pred):
                            dev_result[essay.label.test_taker_id] = {
                                    l1: float(p)
                                    for l1, p in zip(lang_vocab, ps)}
 
            print('DEV', dev_loss, n_batches, time.time()-t0, flush=True)
            if dev_loss < best_dev_loss:
                best_dev_loss = dev_loss
                serializers.save_npz(args.model + '.npz', model)
                with open(args.model + '.metadata', 'wb') as f:
                    pickle.dump((args, vocab, pos_vocab, lang_vocab), f, -1)
                print('SAVE', n_batches, flush=True)

                test_result = {}
                with chainer.using_config('train', False):
                    with chainer.using_config('enable_backprop', False):
                        for i in range(0, len(test_essays), batch_size):
                            batch_essays = test_essays[i:i+batch_size]
                            batch_token, batch_pos = list(zip(*map(
                                encode_essay, batch_essays)))
                            pred = F.softmax(model(batch_token, batch_pos))
                            pred = cuda.to_cpu(pred.data)
                            for essay, ps in zip(batch_essays, pred):
                                test_result[essay.label.test_taker_id] = {
                                        l1: float(p)
                                        for l1, p in zip(lang_vocab, ps)}
                with open(args.model + '.predictions.test', 'wb') as f:
                    pickle.dump(test_result, f, -1)
                with open(args.model + '.predictions.dev', 'wb') as f:
                    pickle.dump(dev_result, f, -1)

            confusion = np.zeros((len(lang_vocab),)*2, dtype=np.int32)
            for x,y in zip(dev_pred, dev_target):
                confusion[x,y] += 1
            print('ACC',
                    sum(x==y for x,y in zip(dev_pred, dev_target))/
                    len(dev_target),
                    n_batches,
                    flush=True)
            print(confusion, flush=True)

        n_batches += 1


if __name__ == '__main__': main()


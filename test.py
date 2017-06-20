#!/usr/bin/env python

from __future__ import print_function
import argparse
import os
import random
from collections import defaultdict
# random.seed(1000)
import numpy as np
# np.random.seed(1000)

import chainer
import chainer.links as L
import chainer.functions as F

from datasets import NLIDataset
from train import CNNModel

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
    print('nli!')
    train = NLIDataset(args.dataset, 'train', char_to_id, label_to_id, word_to_id, pos_to_id, args.maxlen, args.batchsize, repeat=True, subset=args.subset, use_bow=args.use_bow)
    test = NLIDataset(args.dataset, 'dev', char_to_id, label_to_id, word_to_id, pos_to_id, args.maxlen, args.batchsize, repeat=False, shuffle=False, subset=args.subset, use_bow=args.use_bow)
    test = NLIDataset(args.dataset, 'test', char_to_id, label_to_id, word_to_id, pos_to_id, args.maxlen, args.batchsize, repeat=False, shuffle=False, subset=args.subset, use_bow=args.use_bow)

    vocab_size = [len(char_to_id[0]) + 1, len(char_to_id[1]) + 1]#max(map(max, train.pos_dataset))
    word_vocab_size = len(word_to_id) + 1
    n_labels = 2 if len(label_to_id) == 0 else len(label_to_id)-1#Test includes 'X'
    n_pos = len(pos_to_id)

    print('{0}, {1} char ids'.format(vocab_size[0], vocab_size[1]))
    print('{0} word ids'.format(word_vocab_size))
    print('{0} labels'.format(n_labels))
    print('{0} pos'.format(n_pos))
    model = L.Classifier(CNNModel(
        vocab_size, n_labels, word_vocab_size, n_pos, args.out_size, args.dropout, args))

    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        model.to_gpu()
        model.predictor.embed_tri.to_gpu()
        model.predictor.embed_four.to_gpu()

    chainer.serializers.load_npz(args.name, model)

    with chainer.using_config('train', False):
        import pdb; pdb.set_trace()
        batch = test_data.data[i:i+batch_size]
        source = encode_source([src for src,_,_ in batch])
        feats = encode_features([f for _,_,f in batch], dtype=xp.float32)

        states = [model.forward(source, feats) for model in models]
        beam = beam_search(states, args.beam_size,
                    max_length=20+max(len(src) for src,_,_ in batch))

        target = [''.join(alphabet[int(cuda.to_cpu(x))]
                  for x in hypothesis.history[1:-1])
                  for hypothesis in beam]
        if batch[0][1] is not None:
            n_correct += int(batch[0][1] == target[0])
            if batch[0][1] in target:
                n_correct10 += 1
                rr_sum += 1.0 / (1.0 + target.index(batch[0][1]))
            n_total += 1
        for src,trg,f in batch:
            print(src, trg, ';'.join(f), ','.join(target), file=sys.stderr)
            print('\t'.join((src, target[0], ';'.join(f))))

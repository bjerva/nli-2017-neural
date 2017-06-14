'''Preprocess IMDB dataset.'''
from __future__ import (absolute_import, division, print_function, unicode_literals)

import sys
import os
import argparse
import pickle
import random

import numpy as np

from chainer.dataset import dataset_mixin
from chainer.iterators import SerialIterator

from collections import defaultdict


def convert_file(filepath, char_to_id):
    label = 0 if 'pos' in filepath else 1

    with open(filepath) as ifile:
        return [char_to_id[c] for w in ifile.read().split(' ') for c in w]

def discover_dataset(path, char_to_id):
    dataset = []
    for root, _, files in os.walk(path):
        for sfile in [f for f in files if '.txt' in f]:
            filepath = os.path.join(root, sfile)
            dataset.append(convert_file(filepath, char_to_id))
    return dataset

def find_ngrams(string, n):
    return list(zip(*[string[i:] for i in range(n)]))

def read_nli(data_dir, fold, char_to_id, label_to_id, maxlen=128, subset=False):
    unk = char_to_id['<UNK>']
    bos = char_to_id['<S>']
    eos = char_to_id['</S>']

    train_label_path = os.path.join(data_dir, 'labels', fold, 'labels.{0}.csv'.format(fold))
    labels = {}
    with open(train_label_path, 'r') as in_f:
        in_f.readline() # skip csv header
        for line in in_f:
            fields = line.strip().split(',')
            entry_id, native_lang = fields[0], fields[3]
            if subset == 1 and native_lang not in ['FRE', 'JPN', 'TUR']: continue
            if subset == 2 and native_lang not in ['TEL', 'HIN']: continue
            labels[entry_id] = label_to_id[native_lang]

    use_tokenized = False
    version = 'tokenized' if use_tokenized else 'original'
    train_essay_path = os.path.join(data_dir, 'essays', fold, version)
    sents = defaultdict(list)
    for root, dirs, files in os.walk(train_essay_path):
        for idx, fname in enumerate(files):
            if fname[:-4] not in labels: continue
            with open(os.path.join(train_essay_path, fname)) as in_f:
                #for line in in_f:
                #char_rep = [[bos] + [char_to_id[char] for char in word] + [eos] for line in in_f for word in line.split()]
                #for line in in_f:
                #char_rep = [bos] + [char_to_id[ngram] for line in in_f for ngram in find_ngrams(line, 4)+find_ngrams(line, 3)+find_ngrams(line, 2)+find_ngrams(line, 1)] + [eos]
                lines = in_f.readlines()
                char_rep = []
                char_rep.append([bos] + [char_to_id[ngram] for line in lines for ngram in find_ngrams(line, 3)] + [eos])
                char_rep.append([bos] + [char_to_id[ngram] for line in lines for ngram in find_ngrams(line, 4)] + [eos])

                # char_rep = []
                # for line in in_f:
                #     for n in range(6):
                #         char_rep.extend([char_to_id[''.join(ngram)] for ngram in find_ngrams(line, n)])

                if char_rep: # No empty lines
                    sents[fname[:-4]].append(char_rep)


            #if idx > 20:
            #    break
    #import pdb; pdb.set_trace()
    print(len(char_to_id))
    X, y = [], []
    for key, entry in sorted(sents.items(), key=lambda x: -len(x[1])):
        #if labels[key] not in [10, 8]: continue
        for sent in entry:
            X.append(sent)
            y.append(labels[key])

    return X, y


def pad_dataset(dataset, maxlen):
    return np.array(
        [np.pad(r, (0, maxlen-len(r)), mode='constant') if len(r) < maxlen else np.array(r[:maxlen], dtype=np.int32)
         for r in dataset], dtype=np.int32)

#%%
class IMDBDataset(dataset_mixin.DatasetMixin):
    def __init__(self, path, dict_path, char_to_id, maxlen=128):
        pos_path = os.path.join(path, 'pos')
        neg_path = os.path.join(path, 'neg')

        #with open(dict_path, 'rb') as dfile:
        #    wdict = pickle.load(dfile)

        self.pos_dataset = pad_dataset(discover_dataset(pos_path, char_to_id), maxlen).astype('i')
        self.neg_dataset = pad_dataset(discover_dataset(neg_path, char_to_id), maxlen).astype('i')

    def __len__(self):
        return len(self.pos_dataset) + len(self.neg_dataset)

    def get_example(self, i):
        is_neg = i >= len(self.pos_dataset)
        dataset = self.neg_dataset if is_neg else self.pos_dataset
        idx = i - len(self.pos_dataset) if is_neg else i
        label = 0 if is_neg else 1

        return (dataset[idx], np.array(label, dtype=np.int32))

class NLIDataset(SerialIterator):
    def __init__(self, path, fold, char_to_id, label_to_id, maxlen=128, batch_size=1, repeat=True, shuffle=True, subset=False, use_bow=False):
        X, y = read_nli(path, fold, char_to_id, label_to_id, maxlen=maxlen, subset=subset)
        print('{0} instances in {1}'.format(len(X), fold))
        #print('longest sentence is {0} chars'.format(max(map(len, X))))
        self.dataset = list(zip(X, y))#pad_dataset(X, maxlen)
        self.maxlen = maxlen
        self.use_bow = use_bow
        super(NLIDataset, self).__init__(
            self.dataset, batch_size, repeat, shuffle)

    def __next__(self):
        if not self._repeat and self.epoch > 0:
            raise StopIteration

        self._previous_epoch_detail = self.epoch_detail

        i = self.current_position
        i_end = i + self.batch_size
        N = len(self.dataset)

        if self._order is None:
            batch = self.get_data(self.dataset[i:i_end])
        else:
            batch = self.get_data([self.dataset[index] for index in self._order[i:i_end]])

        if i_end >= N:
            if self._repeat:
                rest = i_end - N
                if self._order is not None:
                    np.random.shuffle(self._order)
                if rest > 0:
                    if self._order is None:
                        batch.extend(self.get_data(self.dataset[:rest]))
                    else:
                        batch.extend(self.get_data([self.dataset[index]
                                      for index in self._order[:rest]]))
                self.current_position = rest
            else:
                self.current_position = 0

            self.epoch += 1
            self.is_new_epoch = True
        else:
            self.is_new_epoch = False
            self.current_position = i_end

        return batch

    next = __next__ # This appears to be necessary for some reason...

    def get_data(self, data):
        X, y = zip(*data)

        mlen = self.maxlen#max(map(len, X))
        #mlen_word = 16
        # if max(map(len, X)) > mlen:
        #     print(max(map(len, X)))

        X_trigram  = np.asarray([sent[0][:mlen]  + [-1]*(mlen-len(sent[0])) for sent in X], dtype=np.int32)
        X_fourgram = np.asarray([sent[1][:mlen]  + [-1]*(mlen-len(sent[1])) for sent in X], dtype=np.int32)
        X = np.hstack([X_trigram, X_fourgram])
        if self.use_bow:
            X_onehot = np.zeros((X.shape[0], 140000), dtype=np.float32)
            for idx, sent in enumerate(X):
                for word_id in sent:
                    if word_id == -1:
                        break
                    X_onehot[idx, word_id] += 1

            X = np.asarray(X, dtype=np.float32)
            X = np.hstack([X, X_onehot])
        #
        # X = X_onehot
        # X = [[word[:mlen_word] + [-1]*(mlen_word-len(word)) for word in sent] for sent in X]
        # X = [sent[:mlen] + [[-1]*mlen_word for _ in range(mlen-len(sent))] for sent in X]
        # X = np.asarray(X, dtype=np.int32)
        #import pdb; pdb.set_trace()
        return list(zip(X, np.array(y, dtype=np.int32)))#.reshape(-1, 1)))#list(zip(X_l1_words, X_l2_indices, X_l2_contexts, y))

#%%
# import chainer
# train = IMDBDataset('data/aclImdb/train', 'data/dict.pckl')
# train_iter = chainer.iterators.SerialIterator(train, 16)


# batch = next(train_iter)
# batch[0]

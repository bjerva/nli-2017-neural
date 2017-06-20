from collections import namedtuple, defaultdict
import os.path
import xml.etree.ElementTree as et

LabelItem = namedtuple('LabelItem',
        ['test_taker_id', 'speech_prompt', 'essay_prompt', 'L1'])

class NLIDataset:
    def __init__(self, path, parts=['dev', 'train']):
        self.path = path
        self.labels = {}
        for part in parts:
            self.read_part(part)

    def read_part(self, part):
        if part == 'test':
            label_file = os.path.join(
                    self.path, 'data', 'labels',
                    part, 'essay.labels.%s.csv' % part)
            with open(label_file, 'r') as f:
                next(f)
                lines = [line.rstrip('\n').split(',') for line in f]
                labels = [LabelItem(test_taker_id, 'X', essay_prompt, l1)
                          for test_taker_id, essay_prompt, l1 in lines]
        else:
            label_file = os.path.join(
                    self.path, 'data', 'labels', part, 'labels.%s.csv' % part)
            with open(label_file, 'r') as f:
                next(f)
                labels = [LabelItem(*line.rstrip('\n').split(','))
                          for line in f]

        self.labels[part] = {label.test_taker_id: label for label in labels}

    def essay_path(self, part, processing='original'):
        return os.path.join(self.path, 'data', 'essays', part, processing)

    def essay_filename(self, part, test_taker_id, processing='original'):
        return os.path.join(
                self.essay_path(part, processing),
                test_taker_id + ('.xml' if processing=='parsed' else '.txt'))
    
    def get_sents(self, part, test_taker_id):
        filename = self.essay_filename(part, test_taker_id, 'parsed')
        tree = et.parse(filename)
        for sentence in tree.iter('sentence'):
            tokens = []
            for token in sentence.find('tokens'):
                pos = token.find('POS').text
                word = token.find('word').text
                lemma = token.find('lemma').text
                tokens.append((word, lemma, pos))
            if tokens: yield tokens

    #def get_sents(self, part, test_taker_id, delexicalize=True):
    #    lexical_pos = set(('CD FW JJ JJR JJS LS NN NNS NNP NNPS RB RBR RBS '
    #                       'SYM UH VB VBD VBG VBN VBP VBZ').split())
    #    filename = self.essay_filename(part, test_taker_id, 'parsed')
    #    tree = et.parse(filename)
    #    for sentence in tree.iter('sentence'):
    #        delex_tokens = []
    #        for token in sentence.find('tokens'):
    #            pos = token.find('POS').text
    #            word = token.find('word').text
    #            delex = pos if delexicalize and pos in lexical_pos \
    #                    else word
    #            delex_tokens.append(delex)
    #        yield delex_tokens



if __name__ == '__main__':
    import sys
    import numpy as np

    dataset = NLIDataset(sys.argv[1])
    l1_lengths = defaultdict(list)
    l1_prompt_lengths = defaultdict(list)
    for label in dataset.labels['train'].values():
        with open(dataset.essay_filename('train', label.test_taker_id)) as f:
            length = len(f.read())
            l1_lengths[label.L1].append(length)
            #l1_prompt_lengths[(label.L1, label.essay_prompt)].append(length)

    for l1, lengths in sorted(l1_lengths.items()):
        print(l1, np.mean(lengths), np.median(lengths), np.std(lengths))


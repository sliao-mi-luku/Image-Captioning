"""
COCO Image Captioning - Vocabulary tools
"""

import os
import pickle
from pycocotools.coco import COCO
from collections import Counter
import nltk
nltk.download('punkt')


class Vocabulary():
    """
    Vocabulary object (Originally implemented by Udacity Computer Vision Nanodegree)
    """
    def __init__(self, vocab_threshold, vocab_file="/content/vocab.pkl",
                 start_word="<start>", end_word="<end>", unk_word="<unk>",
                 annotations_file="./opt/cocoapi/annotations/captions_train2014.json",
                 vocab_from_file=False):
        """
        Args
            vocab_threshold: minimum count of the words to be considered a unique token
            vocab_file: vocab file
            start_word: start-of-sentence token
            end_word: end-of-sentence token
            unk_word: unknown-word token
            annotations_file: annotations file (for training dataset)
            vocab_from_file: (boolean) whether or not to use the existing vocab file
        """
        self.vocab_threshold = vocab_threshold
        self.vocab_file = vocab_file
        self.start_word = start_word
        self.end_word = end_word
        self.unk_word = unk_word
        self.annotations_file = annotations_file
        self.vocab_from_file = vocab_from_file
        self.get_vocab()

    def get_vocab(self):
        """
        Load/Create the vocab file
        """
        # load and use the existing vocab file
        if os.path.exists(self.vocab_file) & self.vocab_from_file:
            with open(self.vocab_file, 'rb') as f:
                vocab = pickle.load(f)
                self.word2idx = vocab.word2idx
                self.idx2word = vocab.idx2word
            print('Vocabulary successfully loaded from vocab.pkl file!')
        # build a new vocab file
        else:
            self.build_vocab()
            with open(self.vocab_file, 'wb') as f:
                pickle.dump(self, f)

    def build_vocab(self):
        """
        Create dicts for converting tokens to integers (and vice-versa)
        """
        self.init_vocab()
        self.add_word(self.start_word)
        self.add_word(self.end_word)
        self.add_word(self.unk_word)
        self.add_captions()

    def init_vocab(self):
        """
        Initialize the dictionaries for converting tokens to integers (and vice-versa)
        """
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        """
        Add a token to the vocabulary
        """
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def add_captions(self):
        """
        Loop over training captions and add all tokens to the vocabulary that meet or exceed the threshold
        """
        # initialize coco api
        coco = COCO(self.annotations_file)
        # initialize counter
        counter = Counter()
        # all annotation ids
        ids = coco.anns.keys()
        # iterate over annotation ids
        for i, id in enumerate(ids):
            # retrieve caption
            caption = str(coco.anns[id]['caption'])
            # tokenize
            tokens = nltk.tokenize.word_tokenize(caption.lower())
            # update counter
            counter.update(tokens)
            # print stats
            if i % 100000 == 0:
                print("[%d/%d] Tokenizing captions..." % (i, len(ids)))
        # keep only the words whose count is greater than or equal to seld.vocab_threshold
        words = [word for word, cnt in counter.items() if cnt >= self.vocab_threshold]
        # add qualified words into the vocab dict
        for word in words:
            self.add_word(word)

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx[self.unk_word]
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)

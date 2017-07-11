import numpy as np
from keras.utils import np_utils
import cPickle as pickle
import subprocess

def parse_line(line, mode, max_len, nclasses):
    x_y = line.split('<:>')
    indices = [int(ch) for ch in x_y[0].split(',')]
    indices = np.asarray([0 for _ in xrange(max_len - len(indices))] + indices)
    if mode == 'lm':
        # pad the sequence of characters
        X_c = np.asarray(indices[:-1])
        y_c = np_utils.to_categorical(indices[1:], nclasses)
    elif mode == 'clf':
        X_c = np.asarray(indices)
        y_c = np_utils.to_categorical([int(x_y[1])], nclasses)[0]
    return X_c, y_c

def get_line_count(fname):
    p = subprocess.Popen(['wc', '-l', fname], stdout = subprocess.PIPE,
                                              stderr = subprocess.PIPE)
    result, err = p.communicate()
    if p.returncode != 0:
        raise IOError(err)
    return int(result.strip().split()[0])

class Corpus:
    # in-memory data
    def __init__(self, data_file, mode, max_len, nclasses):
        self.X = []
        self.y = []
        self.read_data(data_file, mode, max_len, nclasses)
        print 'Number of lines in %s: %d' % (data_file, len(self.X))

    def read_data(self, data_file, mode, max_len, nclasses):

        if data_file is None:
            return None

        with open(data_file, 'r') as fh:
            for line in fh:
                X_c, y_c = parse_line(line, mode, max_len, nclasses)
                self.X.append(X_c)
                self.y.append(y_c)

        self.X = np.asarray(self.X)
        self.y = np.asarray(self.y)

class Generator:
    # streaming data
    def __init__(self, data_file, mode, max_len, nclasses):
        self.data_file = data_file
        self.mode = mode
        self.max_len = max_len
        self.nclasses = nclasses
        self.wc = get_line_count(self.data_file)
        print 'Number of lines in %s: %d' % (self.data_file, self.wc)

    def get_mini_batch(self, batch_size):
        # mini-batch generator
        X = []
        y = []
        while True:
            done = False
            with open(self.data_file) as fh:
                for line in fh:
                    X_c, y_c = parse_line(line, self.mode, self.max_len, self.nclasses)
                    X.append(X_c)
                    y.append(y_c)
                    if len(X) == batch_size:
                        done = True
                        X = np.asarray(X)
                        y = np.asarray(y)
                        yield (X, y)
                    if done:
                        done = False
                        X = []
                        y = []

    def get_example(self):
        # mini-batch generator
        # Currently this method will miss the last few examples that do not make up the minibatch i.e last (wc % batch_size) examples
        while True:
            with open(self.data_file) as fh:
                for line in fh:
                    X_c, y_c = parse_line(line, self.mode, self.max_len, self.nclasses)
                    yield (np.asarray([X_c]), np.asarray([y_c]))


class TweetCorpus:

    def __init__(self, train_file = None, val_file = None, test_file = None, unld_train_file = None, unld_val_file = None, dictionaries_file = None):

        self.char2idx, self.label2idx, self.max_len = pickle.load(open(dictionaries_file, "rb"))

        self.idx2char = {v:k for k, v in self.char2idx.iteritems()}
        self.idx2char[0] = ''

        self.tr_data = Corpus(train_file, 'clf', self.max_len, len(self.label2idx))
        self.val_data = Corpus(val_file, 'clf', self.max_len, len(self.label2idx))
        self.te_data = Corpus(test_file, 'clf', self.max_len, len(self.label2idx))
        self.unld_tr_data = Generator(unld_train_file, 'lm', self.max_len, len(self.char2idx) + 1)
        self.unld_val_data = Generator(unld_val_file, 'lm', self.max_len, len(self.char2idx) + 1)
#         self.unld_tr_data = Corpus(unld_train_file, 'lm', self.max_len, len(self.char2idx) + 1)
#         self.unld_val_data = Corpus(unld_val_file, 'lm', self.max_len, len(self.char2idx) + 1)

    def get_data_for_classification(self):
        return self.tr_data.X, self.val_data.X, self.te_data.X, self.tr_data.y, self.val_data.y, self.te_data.y

    def get_data_for_lm(self):
        return self.unld_tr_data.X, self.unld_val_data.X, self.unld_tr_data.y, self.unld_val_data.y


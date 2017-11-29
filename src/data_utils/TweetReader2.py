from keras.utils import np_utils
from sklearn.model_selection import StratifiedKFold
import subprocess

import cPickle as pickle
import numpy as np

def parse_line(line, mode, max_len, nclasses):
    x_y = line.split('<:>')
    indices = [int(ch) for ch in x_y[0].split(',')]
    if mode == 'lm':
        X_c = np.asarray(indices)
        y_c = None
    elif mode == 'clf':
        indices = [0 for _ in xrange(max_len - len(indices))] + indices
        X_c = np.asarray(indices)
        y_c = np_utils.to_categorical([int(x_y[1])], nclasses)
    elif mode == 'seq2seq':
        indices = [0 for _ in xrange(max_len - len(indices))] + indices
        X_c = np.asarray(indices)
        y_c = np_utils.to_categorical(indices, nclasses)
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
        while True:
            with open(self.data_file) as fh:
                for line in fh:
                    X_c, y_c = parse_line(line, self.mode, self.max_len, self.nclasses)
                    yield (np.asarray([X_c]), np.asarray([y_c]))


class TweetCorpus:

    def __init__(self, train_file = None, val_file = None, test_file = None, unld_train_file = None, unld_val_file = None, dictionaries_file = None):

        self.W, self.char2idx, self.label2idx, self.class_weights, self.max_len = pickle.load(open(dictionaries_file, "rb"))

        self.idx2char = {v:k for k, v in self.char2idx.iteritems()}
        self.idx2char[0] = ''

        if train_file is None:
            self.tr_data = None
        else:
            self.tr_data = Corpus(train_file, 'clf', self.max_len, len(self.label2idx))

        if val_file is None:
            self.val_data = None
        else:
            self.val_data = Corpus(val_file, 'clf', self.max_len, len(self.label2idx))

        if test_file is None:
            self.te_data = None
        else:
            self.te_data = Corpus(test_file, 'clf', self.max_len, len(self.label2idx))

        if unld_train_file is None:
            self.unld_tr_data = None
        else:
            # TODO:
            # Uncomment to pre-train a language model
            self.unld_tr_data = Corpus(unld_train_file, 'lm', self.max_len, len(self.char2idx) + 1)
            # self.unld_tr_data = Generator(unld_train_file, 'lm', self.max_len, len(self.char2idx) + 1)
            # Uncomment to pre-train a classifier
            # self.unld_tr_data = Corpus(unld_train_file, 'clf', self.max_len, len(self.label2idx))

        if unld_val_file is None:
            self.unld_val_data = None
        else:
            # TODO:
            # Uncomment to pre-train a language model
            self.unld_val_data = Corpus(unld_val_file, 'lm', self.max_len, len(self.char2idx) + 1)
            # self.unld_val_data = Generator(unld_val_file, 'lm', self.max_len, len(self.char2idx) + 1)
            # Uncomment to pre-train a classifier
            # self.unld_val_data = Corpus(unld_val_file, 'clf', self.max_len, len(self.label2idx))

    def get_data_for_classification(self):
        return self.tr_data.X, self.val_data.X, self.te_data.X, self.tr_data.y, self.val_data.y, self.te_data.y

    def get_data_for_cross_validation(self, folds = 3):
        # combine X_train, X_val and use the combined dataset for cross validation
        X_train = np.concatenate((self.tr_data.X, self.val_data.X), axis = 0)
        y_train = np.concatenate((self.tr_data.y, self.val_data.y), axis = 0)
        skf = StratifiedKFold(n_splits = folds)
        for train_index, test_index in skf.split(X_train, np.argmax(y_train, axis = 1)):
            yield X_train[train_index], X_train[test_index], y_train[train_index], y_train[test_index]

    def flatten(self, X, context_size):
        if X == None:
            return None, None
        X_tcs = []
        y_tcs = []
        for X_c in X:
            # pad atleast context_size - 1 zeros in the beginning
            X_c = X_c.tolist()
            X_c = [0 for _ in xrange(context_size - 1)] + X_c
            s = 0
            while s + context_size < len(X_c):
                X_tcs.append(X_c[s:s + context_size])
                y_tcs.append(X_c[s + context_size])
                s += 1

        X_tcs = np.asarray(X_tcs)
        y_tcs = np.asarray(y_tcs)
        '''
        https://github.com/fchollet/keras/issues/483
        The trick to fix issue with the error expecting a 3D input when using
        sparse_categorical_crossentropy is to format outputs in a sparse 3-dimensional way.
        So instead of formatting the output like this:
        y_indices_naive = [1,5,4300,...]
        is should be formatted this way:
        y_indices_naive = [[1,], [5,] , [4300,],...]
        That will make Keras happy and it'll trained the model as expected.
        '''
        y_tcs = np.expand_dims(y_tcs, 1)
        return X_tcs, y_tcs

    def get_data_for_lm(self, truncate = False, context_size = 10):
        if truncate:
            # After truncation, X will have shape [-1, context_size] and y will have shape [-1,1]
            X_tr, y_tr = self.flatten(self.unld_tr_data.X, context_size)
            X_val, y_val = self.flatten(self.unld_val_data.X, context_size)
            return X_tr, X_val, y_tr, y_val
        else:
            X_tr = self.unld_tr_data.X[:, :-1]
            y_tr = self.unld_tr_data.X[:, 1:]
            y_tr = np.expand_dims(y_tr, 2)
            X_val = self.unld_val_data.X[:, :-1]
            y_val = self.unld_val_data.X[:, 1:]
            y_val = np.expand_dims(y_val, 2)
            return X_tr, X_val, y_tr, y_val

    def get_class_names(self):

        class_names = []
        idx2label = {v:k for k, v in self.label2idx.iteritems()}
        for idx in xrange(len(idx2label)):
            class_names.append(idx2label[idx])

        return class_names

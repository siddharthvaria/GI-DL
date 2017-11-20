import numpy as np
import argparse
from keras.utils import np_utils
from keras_impl.models import Classifier, LanguageModel
from sklearn.metrics import classification_report
from data_utils.TweetReader2 import TweetCorpus
from TernaryClassifier import print_hyper_params

import os

def get_class_weights(y):
    counts = {}

    for y_c in y:
        curr_class = np.argmax(y_c)
        if curr_class in counts.keys():
            counts[curr_class] += 1
        else:
            counts[curr_class] = 1

    total = 0.0
    for cls in counts.keys():
        total += (float(1) / counts[cls])

    K = float(1) / total

    n_classes = len(counts)
    class_weights = {}
    for i in xrange(n_classes):
        class_weights[i] = (K / counts[i])

    return class_weights

def get_binary_data(X, y, pos_classes, neg_classes):

    # returns X,y such that all classes in neg_classes are collapsed to class '0'
    # and all classes in pos_classes are collapsed to class '1'

    if X is None or y is None:
        return None, None

    if pos_classes is None or neg_classes is None:
        return None, None

    _X = []
    _y = []

    for x_c, y_c in zip(X, y):
        curr_class = np.argmax(y_c)
        if curr_class in pos_classes:
            _X.append(x_c)
            _y.append(1)
        elif curr_class in neg_classes:
            _X.append(x_c)
            _y.append(0)

    _X = np.asarray(_X)
    _y = np_utils.to_categorical(_y, 2)
    return _X, _y

def print_classification_reports(corpus, y_test, y_pred1, y_pred2):
    y_pred = []
    for y_c1, y_c2 in zip(y_pred1, y_pred2):
        if y_c1 == 0:
            y_pred.append(corpus.label2idx['other'])
        else:
            if y_c2 == 0:
                y_pred.append(corpus.label2idx['loss'])
            else:
                y_pred.append(corpus.label2idx['aggress'])

    # print classification report1
    y_test1 = []
    for y_c in y_test:
        curr_class = np.argmax(y_c)
        if curr_class == corpus.label2idx['other']:
            y_test1.append(0)
        else:
            y_test1.append(1)

    print 'Classification report of classifier 1: '
    print classification_report(y_test1, y_pred1, target_names = ['other', 'aggress/loss'])
    # print classification report2
    y_test2 = []
    _y_pred2 = []
    for y_c1, y_c2 in zip(y_test, y_pred2):
        curr_class = np.argmax(y_c1)
        if curr_class == corpus.label2idx['aggress']:
            y_test2.append(1)
            _y_pred2.append(y_c2)
        elif curr_class == corpus.label2idx['loss']:
            y_test2.append(0)
            _y_pred2.append(y_c2)

    print 'Classification report of classifier 2: '
    print classification_report(y_test2, _y_pred2, target_names = ['loss', 'aggress'])
    # print final classification report
    print 'Overall classification report: '
    print classification_report(np.argmax(y_test, axis = 1), y_pred, target_names = corpus.get_class_names())

def train_cc_classifier(clf1, clf2, corpus, args):
    X_train, X_val, X_test, y_train, y_val, y_test = corpus.get_data_for_classification()
    # clf1 for loss,aggression v/s others
    X_train1, y_train1 = get_binary_data(X_train, y_train, [corpus.label2idx['loss'], corpus.label2idx['aggress']], [corpus.label2idx['other']])
    X_val1, y_val1 = get_binary_data(X_val, y_val, [corpus.label2idx['loss'], corpus.label2idx['aggress']], [corpus.label2idx['other']])
    # X_test1, y_test1 = get_binary_data(X_test, y_test, [corpus.label2idx['loss'], corpus.label2idx['aggress']], [corpus.label2idx['other']])
    y_pred1 = clf1.fit(X_train1, X_val1, X_test, y_train1, y_val1, get_class_weights(y_train1), args)
    # clf2 for aggression v/s loss
    X_train2, y_train2 = get_binary_data(X_train, y_train, [corpus.label2idx['aggress']], [corpus.label2idx['loss']])
    X_val2, y_val2 = get_binary_data(X_val, y_val, [corpus.label2idx['aggress']], [corpus.label2idx['loss']])
    # X_test2, y_test2 = get_binary_data(X_test, y_test, [corpus.label2idx['aggress']], [corpus.label2idx['loss']])
    y_pred2 = clf2.fit(X_train2, X_val2, X_test, y_train2, y_val2, get_class_weights(y_train2), args)
    # print classification_report(np.argmax(y_test1, axis = 1), y_pred1, target_names = ['loss', 'aggress'])
    print_classification_reports(corpus, y_test, y_pred1, y_pred2)

def parse_arguments():

    parser = argparse.ArgumentParser(description = '')

    parser.add_argument('train_file', type = str)
    parser.add_argument('val_file', type = str)
    parser.add_argument('test_file', type = str)
    parser.add_argument('dictionaries_file', type = str)
    parser.add_argument('model_save_dir', type = str)
    parser.add_argument('mode', type = str)
    parser.add_argument('--pretrain', type = bool, default = False)
    parser.add_argument('--unld_train_file', type = str, default = None)
    parser.add_argument('--unld_val_file', type = str, default = None)
    parser.add_argument('--n_epochs', type = int, default = 50)
    parser.add_argument('--lstm_hidden_dim', type = int, default = 256)
    parser.add_argument('--dense_hidden_dim', type = int, default = 256)
    parser.add_argument('--emb_dim', type = int, default = 128)
    parser.add_argument('--dropout', type = float, default = 0.5)
    parser.add_argument('--batch_size', type = int, default = 64)

    args = vars(parser.parse_args())

    return args

def main(args):

    # corpus = TweetCorpus(unlabeled_tweets_file = args['tweets_file'])
    # corpus = TweetCorpus(args['train_file'], args['val_file'], args['test_file'], args['tweets_file'])
    corpus = TweetCorpus(args['train_file'], args['val_file'], args['test_file'], args['unld_train_file'], args['unld_val_file'], args['dictionaries_file'])

    args['max_seq_len'] = corpus.max_len
    args['nclasses'] = 2
    args['nchars'] = len(corpus.char2idx) + 1

    print_hyper_params(args)

    if args['mode'] == 'lm':
        print 'Creating language model . . .'
        lm = LanguageModel(args)
        print 'Training language model . . .'
        # train_lm(lm, corpus, args)
        lm.fit(corpus, args)
    elif args['mode'] == 'clf':
        print 'Creating classifier model . . .'
        clf1 = Classifier(args)
        clf2 = Classifier(args)
        # if the weights from the lm exists then use those weights instead
        if args['pretrain'] and os.path.isfile(os.path.join(args['model_save_dir'], 'language_model.h5')):
            print 'Loading weights from trained language model . . .'
            clf1.model.load_weights(os.path.join(args['model_save_dir'], 'language_model.h5'), by_name = True)
            clf2.model.load_weights(os.path.join(args['model_save_dir'], 'language_model.h5'), by_name = True)

        print 'Training classifier model . . .'
        train_cc_classifier(clf1, clf2, corpus, args)

if __name__ == '__main__':
    main(parse_arguments())

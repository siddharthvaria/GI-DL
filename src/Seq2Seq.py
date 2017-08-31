from data_utils.TweetReader2 import TweetCorpus
from keras_impl.models import Classifier, AutoEncoder_CNN
from keras_impl.models import AutoEncoder
from sklearn.metrics import classification_report

import numpy as np
from random import randint
import argparse
import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def generate_text(model, corpus, args):

    tr_data_X = corpus.tr_data.X

    input_seqs = [randint(0, len(tr_data_X) - 1) for _ in xrange(10)]

    X_in = tr_data_X[input_seqs]

    X_out = model.predict(X_in, batch_size = len(X_in), verbose = 0)
    print X_out.shape
    X_out = np.argmax(X_out, axis = 2)

    for x_in, x_out in zip(X_in, X_out):

        print '#########################################################'
        print(''.join([corpus.idx2char[idx].encode('utf8') for idx in x_in]))
        print(''.join([corpus.idx2char[idx].encode('utf8') for idx in x_out]))
        print '#########################################################'

def main(args):

    # corpus = TweetCorpus(unlabeled_tweets_file = args['tweets_file'])
    # corpus = TweetCorpus(args['train_file'], args['val_file'], args['test_file'], args['tweets_file'])
    corpus = TweetCorpus(args['train_file'], args['val_file'], args['test_file'], args['unld_train_file'], args['unld_val_file'], args['dictionaries_file'])

    args['max_seq_len'] = corpus.max_len
    args['nclasses'] = len(corpus.label2idx)
    args['nchars'] = len(corpus.char2idx) + 1

    if args['mode'] == 'seq2seq':
        print 'Creating Autoencoder model . . .'
        # ae = AutoEncoder(corpus.W, args)
        ae = AutoEncoder_CNN(corpus.W, args)
        print 'Training Autoencoder model . . .'
        ae.fit(corpus, args)
        if os.path.isfile(os.path.join(args['model_save_dir'], 'autoencoder_model.h5')):
            print 'Loading weights from trained autoencoder model . . .'
            ae.model.load_weights(os.path.join(args['model_save_dir'], 'autoencoder_model.h5'), by_name = True)
        else:
            print 'No trained autoencoder model available . . .!'
            sys.exit(0)
        generate_text(ae.model, corpus, args)
    elif args['mode'] == 'clf':
        print 'Creating classifier model . . .'
        clf = Classifier(corpus.W, args)
        # if the weights from the autoencoder exists then use those weights instead
        if args['pretrain']  and os.path.isfile(os.path.join(args['model_save_dir'], 'autoencoder_model.h5')):
            print 'Loading weights from trained autoencoder model . . .'
            clf.model.load_weights(os.path.join(args['model_save_dir'], 'autoencoder_model.h5'), by_name = True)
        print 'Training classifier model . . .'
        X_train, X_val, X_test, y_train, y_val, y_test = corpus.get_data_for_classification()
        y_pred = clf.fit(X_train, X_val, X_test, y_train, y_val, corpus.class_weights, args)
        print classification_report(np.argmax(y_test, axis = 1), y_pred, target_names = corpus.get_class_names())

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
    parser.add_argument('--nfeature_maps', type = int, default = 256)
    parser.add_argument('--dense_hidden_dim', type = int, default = 512)
    parser.add_argument('--dropout', type = float, default = 0.5)
    parser.add_argument('--batch_size', type = int, default = 32)

    args = vars(parser.parse_args())

    return args

if __name__ == '__main__':

    main(parse_arguments())

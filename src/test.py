import argparse
import codecs
import json
import os
import re
from sklearn.metrics import classification_report

import cPickle as pickle
from data_utils.TweetReader2 import Corpus, add_pad_token
from data_utils.utils import get_delimiter, unicode_csv_reader2, parse_line, datum_to_string, delete_files
from keras_impl.models import LSTMClassifier, CNNClassifier
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def load_args(args):
    _ts = re.search(r'\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2}', args['trained_model']).group(0)
    fpath, _ = os.path.split(args['trained_model'])
    with open(os.path.join(fpath, 'args_' + _ts + '.json'), 'r') as fh:
        _args = json.load(fh)
    for k in args.keys():
        _args[k] = args[k]
    return _args

def update_token2idx(tweet, token2idx, word_level = False):

    if tweet is None:
        return None

    idx = []
    if word_level:
        for token in tweet:
            if token in token2idx:
                idx.append(str(token2idx[token]))
            else:
                idx.append(str(token2idx['__UNK__']))
    else:
        index = 0
        while index < len(tweet):
            if tweet[index:index + len('__URL__')] == '__URL__':
                if '__URL__' in token2idx:
                    idx.append(str(token2idx['__URL__']))
                else:
                    idx.append(str(token2idx['__UNK__']))
                index = index + len('__URL__')
            elif tweet[index:index + len('__USER_HANDLE__')] == '__USER_HANDLE__':
                if '__USER_HANDLE__' in token2idx:
                    idx.append(str(token2idx['__USER_HANDLE__']))
                else:
                    idx.append(str(token2idx['__UNK__']))
                index = index + len('__USER_HANDLE__')
            elif tweet[index:index + len('__RT__')] == '__RT__':
                if '__RT__' in token2idx:
                    idx.append(str(token2idx['__RT__']))
                else:
                    idx.append(str(token2idx['__UNK__']))
                index = index + len('__RT__')
            else:
                ch = tweet[index]
                if ch in token2idx:
                    idx.append(str(token2idx[ch]))
                else:
                    idx.append(str(token2idx['__UNK__']))
                index += 1
    return idx

def read_data(args, token2idx, label2idx, data_file, text_column, label_column, tweet_id_column, max_len, encoding = 'utf8'):

    if data_file is None:
        return

    fpath, fname = os.path.split(data_file)
    dot_index = fname.rindex('.')
    fname_wo_ext = fname[:dot_index]

    indices_file = os.path.join(fpath, fname_wo_ext + '.txt')
    dropped_tweets_file = os.path.join(fpath, fname_wo_ext + '_' + '_dropped.txt')
    delimiter = get_delimiter(data_file)
    with open(data_file, 'r') as fhr, open(indices_file, 'w') as fhw1, codecs.open(dropped_tweets_file, 'w', encoding = encoding) as fhw2:
        reader = unicode_csv_reader2(fhr, encoding, delimiter = delimiter)
        for row in reader:
            # optional parameters to the parser
            # stop_chars = None, normalize = False, add_ss_markers = False, word_level = False
            # line, text_column, label_column, tweet_id_column, max_len, stop_chars = None, normalize = False, add_ss_markers = False, word_level = False
            X_c, y_c, tweet_id = parse_line(row, text_column, label_column, tweet_id_column, max_len, normalize = True, word_level = args['word_level'])
            if X_c is None:
                fhw2.write(tweet_id)
                fhw2.write('\n')
                continue
            X_ids = update_token2idx(X_c, token2idx, word_level = False)
            if y_c is not None:
                y_id = str(label2idx[y_c])
            else:
                y_id = ''

            fhw1.write(datum_to_string(X_ids, y_id, tweet_id))
            fhw1.write('\n')

    return indices_file

def get_class_names(label2idx):

    class_names = []
    idx2label = {v:k for k, v in label2idx.iteritems()}
    for idx in xrange(len(idx2label)):
        class_names.append(idx2label[idx])

    return class_names

def main(args):

    # TODO: change this file to make it more robust
    args = load_args(args)
    W, token2idx, label2idx, _, _, max_len = pickle.load(open(args['dictionaries_file'], "rb"))
    indices_file = read_data(args, token2idx, label2idx, args['test_file'], 'text', 'label', 'tweet_id', max_len)
    corpus = Corpus(indices_file, 'clf')
    X_te = add_pad_token(corpus.X, token2idx['PAD'], max_len)
    if args['arch_type'] == 'lstm':
        print 'Creating LSTM classifier model . . .'
        clf = LSTMClassifier(W, args)
    else:
        print 'Creating CNN classifier model . . .'
        clf = CNNClassifier(W, args)

    fpath, fname = os.path.split(indices_file)
    dot_index = fname.rindex('.')
    fname_wo_ext = fname[:dot_index]

    preds = clf.predict(X_te, args['trained_model'], args['batch_size'])
    print classification_report(corpus.y, np.argmax(preds, axis = 1), target_names = get_class_names(label2idx))
    pickle.dump([corpus.y, np.argmax(preds, axis = 1), preds, get_class_names(label2idx)], open(os.path.join(fpath, fname_wo_ext + '_predictions.p'), 'wb'))
    # delete unwanted file
    delete_files([indices_file])

def parse_arguments():

    parser = argparse.ArgumentParser(description = '')
    parser.add_argument('-tst', '--test_file', type = str, required = True, help = 'labeled test file')
    parser.add_argument('-wl', '--word_level', type = bool, default = False, help = 'If True, tweets will be processed at word level otherwise at char level')
    parser.add_argument('-tm', '--trained_model', type = str, required = True, default = None, help = 'Path to trained model file')
    args = vars(parser.parse_args())

    return args

if __name__ == '__main__':
    main(parse_arguments())

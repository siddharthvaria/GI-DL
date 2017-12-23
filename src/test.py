import argparse
import json
import os
import re
from sklearn.metrics import classification_report

import cPickle as pickle
from data_utils.TweetReader2 import TweetCorpus
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

def main(args):

    args = load_args(args)
    corpus = TweetCorpus(args['arch_type'], None, None, args['test_file'], None, None, args['dictionaries_file'])
    if args['arch_type'] == 'lstm':
        print 'Creating LSTM classifier model . . .'
        clf = LSTMClassifier(corpus.W, args)
    else:
        # args['kernel_sizes'] = [1, 2, 3, 4, 5]
        print 'Creating CNN classifier model . . .'
        clf = CNNClassifier(corpus.W, args)

    preds = clf.predict(corpus.te_data.X, args['trained_model'], args['batch_size'])
    print classification_report(np.argmax(corpus.te_data.y, axis = 1), np.argmax(preds, axis = 1), target_names = corpus.get_class_names())
    pickle.dump([np.argmax(corpus.te_data.y, axis = 1), np.argmax(preds, axis = 1), preds, corpus.get_class_names()], open(os.path.join(args['model_save_dir'], 'test_prediction_' + args['ts'] + '.p'), 'wb'))

def parse_arguments():

    parser = argparse.ArgumentParser(description = '')
    requiredArgs = parser.add_argument_group('required arguments')
    requiredArgs.add_argument('-tst', '--test_file', type = str, required = True, help = 'labeled test file')
    requiredArgs.add_argument('-tm', '--trained_model', type = str, default = None, help = 'Path to trained model file')
    args = vars(parser.parse_args())

    return args

if __name__ == '__main__':
    main(parse_arguments())

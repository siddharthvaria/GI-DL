from data_utils.TweetReader2 import TweetCorpus
from keras_impl.models import build_clf_model, build_lm_model
from keras_impl.models import train_lm_streaming, train_classifier
from sklearn.metrics import classification_report

import numpy as np
import argparse
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def sample(preds, temperature = 1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def generate_text(model, corpus, args):

    start_seqs = ['if u Lookin ', 'RT ', 'I Dnt Carry ', '@younggodumb: ', 'Pockets heavy ', 'I Jst Got ']

    for diversity in [0.2, 0.5, 1.0, 1.2]:

        print '##############################################'

        for ii in xrange(len(start_seqs)):

            start_seq = start_seqs[ii]
            curr_sample = [corpus.char2idx[ch] for ch in start_seq]

            for _ in xrange(np.random.randint(80, 120)):
                seq_in = np.asarray([0 for _ in xrange(args['max_seq_len'] - 1 - len(curr_sample))] + curr_sample)
                inp = np.asarray([seq_in])
                prediction = model.predict(inp, verbose = 0)
                index = sample(prediction[0][-1], diversity)
                # index = np.argmax(prediction[0][-1])
                curr_sample.append(index)

            print(''.join([corpus.idx2char[idx] for idx in curr_sample]))

        print '##############################################'

# def get_emb_matrix(corpus, emb_dim = 300):
#
#     emb_matrix = np.zeros((len(corpus.char2idx) + 1, emb_dim))
#
#     for i in xrange(1, emb_matrix.shape[0]):
#         emb_matrix[i] = np.random.uniform(-0.25, 0.25, emb_dim)
#
#     return emb_matrix

def print_hyper_params(args):

    print 'max_seq_len: ', args['max_seq_len']
    print 'nclasses: ', args['nclasses']
    print 'nchars: ', args['nchars']
    print 'n_epochs: ', args['n_epochs']
    print 'lstm_hidden_dim: ', args['lstm_hidden_dim']
    print 'emb_dim: ', args['emb_dim']
    print 'dropout: ', args['dropout']
    print 'batch_size: ', args['batch_size']

def main(args):

    # corpus = TweetCorpus(unlabeled_tweets_file = args['tweets_file'])
    # corpus = TweetCorpus(args['train_file'], args['val_file'], args['test_file'], args['tweets_file'])
    corpus = TweetCorpus(args['train_file'], args['val_file'], args['test_file'], args['unld_train_file'], args['unld_val_file'], args['dictionaries_file'])

    args['max_seq_len'] = corpus.max_len
    args['nclasses'] = len(corpus.label2idx)
    args['nchars'] = len(corpus.char2idx) + 1

    print_hyper_params(args)

    if args['mode'] == 'lm':
        print 'Creating language model . . .'
        lm = build_lm_model(args)
        print 'Training language model . . .'
        # train_lm(lm, corpus, args)
        train_lm_streaming(lm, corpus, args)
#         if os.path.isfile(os.path.join(args['model_save_dir'], 'language_model.h5')):
#             print 'Loading weights from trained language model for text generation . . .'
#             lm.load_weights(os.path.join(args['model_save_dir'], 'language_model.h5'), by_name = True)
#         else:
#             print 'No trained language model available . . .!'
#             sys.exit(0)
#         print 'Generating some fresh text . . .'
#         generate_text(lm, corpus, args)

    elif args['mode'] == 'clf':
        print 'Creating classifier model . . .'
        clf = build_clf_model(args)
        # if the weights from the lm exists then use those weights instead
        if args['pretrain']  and os.path.isfile(os.path.join(args['model_save_dir'], 'language_model.h5')):
            print 'Loading weights from trained language model . . .'
            clf.load_weights(os.path.join(args['model_save_dir'], 'language_model.h5'), by_name = True)
        print 'Training classifier model . . .'
        X_train, X_val, X_test, y_train, y_val, y_test = corpus.get_data_for_classification()
        y_pred = train_classifier(clf, X_train, X_val, X_test, y_train, y_val, corpus.class_weights, args)
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
    parser.add_argument('--dense_hidden_dim', type = int, default = 256)
    parser.add_argument('--emb_dim', type = int, default = 128)
    parser.add_argument('--dropout', type = float, default = 0.5)
    parser.add_argument('--batch_size', type = int, default = 64)

    args = vars(parser.parse_args())

    return args

if __name__ == '__main__':

    main(parse_arguments())

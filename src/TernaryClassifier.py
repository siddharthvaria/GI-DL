import argparse
import datetime
import os
from sklearn.manifold import TSNE
from sklearn.metrics import classification_report

import cPickle as pickle
from data_utils.TweetReader2 import TweetCorpus
from keras_impl.models import LSTMClassifier, LSTMLanguageModel
import matplotlib.pyplot as plt
import numpy as np


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
    print 'trainable: ', args['trainable']

def vizualize_embeddings(emb_matrix, char2idx):

    assert len(emb_matrix) == (len(char2idx) + 1), 'len(emb_matrix) != len(char2idx) + 1'

    unicode_chars = []
    unicode_embs = []
    for i, ch in enumerate(char2idx.keys()):
        print i, ch
        try:
            ch.encode('ascii')
        except UnicodeEncodeError:
            unicode_chars.append(ch)
            unicode_embs.append(emb_matrix[i + 1])

    unicode_embs = np.asarray(unicode_embs)
    model = TSNE(n_components = 2, random_state = 0)
    np.set_printoptions(suppress = True)
    trans = model.fit_transform(unicode_embs)

    # plot the emoji using TSNE
    fig = plt.figure()
    ax = fig.add_subplot(111)
#     tsne = man.TSNE(perplexity = 50, n_components = 2, init = 'random', n_iter = 300000, early_exaggeration = 1.0,
#                     n_iter_without_progress = 1000)
#     trans = tsne.fit_transform(V)
    x, y = zip(*trans)
    plt.scatter(x, y, marker = 'o', alpha = 0.0)

    for i in range(len(trans)):
        ax.annotate(unicode_chars[i], xy = trans[i], textcoords = 'data')

    plt.grid()
    plt.show()

def main(args):

    # corpus = TweetCorpus(unlabeled_tweets_file = args['tweets_file'])
    # corpus = TweetCorpus(args['train_file'], args['val_file'], args['test_file'], args['tweets_file'])
    corpus = TweetCorpus(args['train_file'], args['val_file'], args['test_file'], args['unld_train_file'], args['unld_val_file'], args['dictionaries_file'])

    args['max_seq_len'] = corpus.max_len
    args['nclasses'] = len(corpus.label2idx)
    args['nchars'] = len(corpus.char2idx) + 1

    # check if W is one hot or dense
    if corpus.W[0][0] == 1:
        args['trainable'] = False
    else:
        args['trainable'] = True

    ts = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    args['ts'] = ts

    print_hyper_params(args)

    if args['mode'] == 'lm':
        print 'Creating language model . . .'
        lm = LSTMLanguageModel(corpus.W, args)
        print 'Training language model . . .'
        # train_lm(lm, corpus, args)
        lm.fit(corpus, args)
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
        clf = LSTMClassifier(corpus.W, args)
#         W_old = corpus.W
        # if the weights from the lm exists then use those weights instead
        if args['pretrain']  and os.path.isfile(os.path.join(args['model_save_dir'], 'language_model.h5')):
            print 'Loading weights from trained language model . . .'
            clf.model.load_weights(os.path.join(args['model_save_dir'], 'language_model.h5'), by_name = True)

        # make sure that Keras is replacing the embeddings with trained embeddings
#         W_new = clf.embedding_layer.get_weights()[0]
#         print 'shape(W_new): ', W_new.shape
#         print 'np.sum: ', np.sum(W_old - W_new)  # np.sum should be much larger than 1

        print 'Training classifier model . . .'
        X_train, X_val, X_test, y_train, y_val, y_test = corpus.get_data_for_classification()
        y_pred = clf.fit(X_train, X_val, X_test, y_train, y_val, corpus.class_weights, args)
        print classification_report(np.argmax(y_test, axis = 1), y_pred, target_names = corpus.get_class_names())
        pickle.dump([np.argmax(y_test, axis = 1), y_pred, corpus.get_class_names()], open(os.path.join(args['model_save_dir'], 'best_prediction_' + args['ts'] + '.p'), 'wb'))
    elif args['mode'] == 'clf_cv':
        # perform cross validation
        y_pred_all = []
        y_all = []
        fold = 1
        for X_train, X_val, y_train, y_val in corpus.get_data_for_cross_validation(3):
            print 'Processing fold:', fold
            fold += 1
            clf = LSTMClassifier(corpus.W, args)
            # if the weights from the lm exists then use those weights instead
            if args['pretrain']  and os.path.isfile(os.path.join(args['model_save_dir'], 'language_model.h5')):
                print 'Loading weights from trained language model . . .'
                clf.model.load_weights(os.path.join(args['model_save_dir'], 'language_model.h5'), by_name = True)
            print 'Training classifier model . . .'
            y_pred = clf.fit(X_train, X_val, X_val, y_train, y_val, corpus.class_weights, args)
            y_pred_all.extend(y_pred)
            y_all.extend(y_val)
        print classification_report(np.argmax(y_all, axis = 1), y_pred_all, target_names = corpus.get_class_names())
        pickle.dump([np.argmax(y_all, axis = 1), y_pred_all, corpus.get_class_names()], open(os.path.join(args['model_save_dir'], 'best_prediction_' + args['ts'] + '.p'), 'wb'))
    elif args['mode'] == 'analyze':
        print 'Analyzing embeddings . . .'
        # lm = LanguageModel(args)
        clf = LSTMClassifier(args)
        clf.model.load_weights(os.path.join(args['model_save_dir'], 'classifier_model.h5'), by_name = True)
        emb_matrix = clf.embedding_layer.get_weights()[0]
        vizualize_embeddings(emb_matrix, corpus.char2idx)

def parse_arguments():

    parser = argparse.ArgumentParser(description = '')
    # even though short flags can be used in the command line, they can not be used to access the value of the arguments
    # i.e args['pt'] will give KeyError.
    requiredArgs = parser.add_argument_group('required arguments')
    requiredArgs.add_argument('-tr', '--train_file', type = str, required = True, help = 'labeled train file')
    requiredArgs.add_argument('-val', '--val_file', type = str, required = True, help = 'labeled validation file')
    requiredArgs.add_argument('-tst', '--test_file', type = str, required = True, help = 'labeled test file')
    requiredArgs.add_argument('-dict', '--dictionaries_file', type = str, required = True, help = 'pickled dictionary file')
    requiredArgs.add_argument('-sdir', '--model_save_dir', type = str, required = True, help = 'directory where trained model should be saved')
    requiredArgs.add_argument('-md', '--mode', type = str, required = True, help = 'mode (clf,clf_cv,lm)')
    requiredArgs.add_argument('-ct', '--clf_type', type = str, required = True, help = 'when mode is clf or clf_cv, clf_type (lstm,cnn) indicates the type of classifier to use')
    parser.add_argument('-pt', '--pretrain', type = bool, default = False)
    parser.add_argument('-unld_tr', '--unld_train_file', type = str, default = None)
    parser.add_argument('-unld_val', '--unld_val_file', type = str, default = None)
    parser.add_argument('-epochs', '--n_epochs', type = int, default = 50)
    parser.add_argument('-lstm_hd', '--lstm_hidden_dim', type = int, default = 256)
    parser.add_argument('-dense_hd', '--dense_hidden_dim', type = int, default = 256)
    parser.add_argument('-edim', '--emb_dim', type = int, default = 128)
    parser.add_argument('-do', '--dropout', type = float, default = 0.5)
    parser.add_argument('-bsz', '--batch_size', type = int, default = 64)

    args = vars(parser.parse_args())

    return args

if __name__ == '__main__':

    main(parse_arguments())

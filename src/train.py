import argparse
import datetime
import json
import os
from sklearn.manifold import TSNE
from sklearn.metrics import classification_report

import cPickle as pickle
from data_utils.TweetReader2 import TweetCorpus
from keras_impl.models import LSTMClassifier, LSTMLanguageModel, CNNClassifier
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
            curr_sample = [corpus.token2idx[tk] for tk in start_seq]

            for _ in xrange(np.random.randint(80, 120)):
                seq_in = np.asarray([0 for _ in xrange(args['max_seq_len'] - 1 - len(curr_sample))] + curr_sample)
                inp = np.asarray([seq_in])
                prediction = model.predict(inp, verbose = 0)
                index = sample(prediction[0][-1], diversity)
                # index = np.argmax(prediction[0][-1])
                curr_sample.append(index)

            print(''.join([corpus.idx2char[idx] for idx in curr_sample]))

        print '##############################################'


def write_args_json(args):
    with open(os.path.join(args['model_save_dir'], 'args_' + args['ts'] + '.json'), 'w') as fh:
        fh.write(json.dumps(args, indent = 4))


def vizualize_embeddings(emb_matrix, token2idx):

    assert len(emb_matrix) == (len(token2idx)), 'len(emb_matrix) != len(token2idx)'

    unicode_chars = []
    unicode_embs = []
    for i, ch in enumerate(token2idx.keys()):
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


def load_corpus(args):

    if args == None:
        return None

    corpus = TweetCorpus(args['train_file'], args['val_file'], args['val_file'], args['unld_train_file'], args['unld_val_file'], args['dictionaries_file'])

    args['max_seq_len'] = corpus.max_len
    args['nclasses'] = len(corpus.label2idx)
    args['ntokens'] = len(corpus.token2idx)

    if args['arch_type'] == 'cnn':
        args['kernel_sizes'] = [1, 2, 3, 4, 5]

    # check if W is one hot or dense
    if corpus.W[0][0] == 1:
        args['trainable'] = False
    else:
        args['trainable'] = True

    ts = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    args['ts'] = ts

    return corpus


def get_output_fname(input_file, suffix):
    fpath, fname = os.path.split(input_file)
    dot_index = fname.rindex('.')
    fname_wo_ext = fname[:dot_index]
    return os.path.join(fpath, fname_wo_ext + '_' + suffix)


def main(args):

    corpus = load_corpus(args)

    # save the arguments for reference
    write_args_json(args)

    if args['mode'] == 'lm':
        if args['arch_type'] == 'lstm':
            # set the context size for lstm language model
            print 'Creating lstm language model . . .'
            lm = LSTMLanguageModel(corpus.W, args)
            print 'Training language model . . .'
            lm.fit(corpus, args)
            pickle.dump([lm.embedding_layer.get_weights()], open(get_output_fname(args['val_file'], 'embeddings.p'), 'wb'))
    elif args['mode'] == 'clf':
        if args['arch_type'] == 'lstm':
            print 'Creating LSTM classifier model . . .'
            clf = LSTMClassifier(corpus.W, args)
            # if the weights from the lm exists then use those weights instead
            if args['trained_model'] is not None and os.path.isfile(args['trained_model']):
                print 'Loading weights from trained language model . . .'
                clf.model.load_weights(args['trained_model'], by_name = True)
        else:
            # args['kernel_sizes'] = [1, 2, 3, 4, 5]
            print 'Creating CNN classifier model . . .'
            clf = CNNClassifier(corpus.W, args)
            # if the weights from the pre-trained cnn exists then use those weights instead
            if args['trained_model'] is not None and os.path.isfile(args['trained_model']):
                print 'Loading weights from trained CNN model . . .'
                clf.model.load_weights(args['trained_model'], by_name = True)

        print 'Training classifier model . . .'
        X_train, X_val, X_test, y_train, y_val, y_test = corpus.get_data_for_classification()
        # preds are output class probabilities
        preds, representations = clf.fit(X_train, X_val, X_test, y_train, y_val, corpus.class_weights, args)
        print classification_report(y_test, np.argmax(preds, axis = 1), target_names = corpus.get_class_names(), digits = 4)
        pickle.dump([y_test, np.argmax(preds, axis = 1), preds, representations, corpus.get_class_names()],
                    open(get_output_fname(args['val_file'], 'predictions.p'), 'wb'))
    elif args['mode'] == 'clf_cv':
        # perform cross validation
        preds_all = []
        y_all = []
        fold = 1
        for X_train, X_val, y_train, y_val in corpus.get_data_for_cross_validation(3):
            print 'Processing fold:', fold
            fold += 1
            if args['arch_type'] == 'lstm':
                print 'Creating LSTM classifier model . . .'
                clf = LSTMClassifier(corpus.W, args)
                # if the weights from the lm exists then use those weights instead
                if args['trained_model'] is not None and os.path.isfile(args['trained_model']):
                    print 'Loading weights from trained language model . . .'
                    clf.model.load_weights(args['trained_model'], by_name = True)
            else:
                # args['kernel_sizes'] = [1, 2, 3, 4, 5]
                print 'Creating CNN classifier model . . .'
                clf = CNNClassifier(corpus.W, args)
                # if the weights from the pre-trained cnn exists then use those weights instead
                if args['trained_model'] is not None and os.path.isfile(args['trained_model']):
                    print 'Loading weights from trained CNN model . . .'
                    clf.model.load_weights(args['trained_model'], by_name = True)
            print 'Training classifier model . . .'
            preds, _ = clf.fit(X_train, X_val, X_val, y_train, y_val, corpus.class_weights, args)
            preds_all.extend(preds)
            y_all.extend(y_val)
        print classification_report(np.argmax(y_all, axis = 1), np.argmax(preds_all, axis = 1), target_names = corpus.get_class_names(), digits = 4)
        pickle.dump([np.argmax(y_all, axis = 1), np.argmax(preds_all, axis = 1), preds_all, corpus.get_class_names()], open(os.path.join(args['model_save_dir'], 'best_prediction_' + args['ts'] + '.p'), 'wb'))
    elif args['mode'] == 'analyze':
        print 'Analyzing embeddings . . .'
        # lm = LanguageModel(args)
        clf = LSTMClassifier(args)
        clf.model.load_weights(os.path.join(args['model_save_dir'], 'classifier_model.h5'), by_name = True)
        emb_matrix = clf.embedding_layer.get_weights()[0]
        vizualize_embeddings(emb_matrix, corpus.token2idx)


def parse_arguments():

    parser = argparse.ArgumentParser(description = '')
    # even though short flags can be used in the command line, they can not be used to access the value of the arguments
    # i.e args['pt'] will give KeyError.
    parser.add_argument('-tr', '--train_file', type = str, default = None, help = 'labeled train file')
    parser.add_argument('-val', '--val_file', type = str, default = None, help = 'labeled validation file')
    parser.add_argument('-dict', '--dictionaries_file', type = str, default = None, help = 'pickled dictionary file (run preprocess_tweets.py to generate the dictionary file)')
    parser.add_argument('-sdir', '--model_save_dir', type = str, default = None, help = 'directory where trained model should be saved')
    parser.add_argument('-md', '--mode', type = str, default = 'clf', help = 'mode (clf,clf_cv,lm)')
    parser.add_argument('-at', '--arch_type', type = str, default = 'cnn', help = 'Type of architecture (lstm,cnn)')
    parser.add_argument('-tm', '--trained_model', type = str, default = None, help = 'Path to trained model file. If provided, training will be continued from this model.')
    parser.add_argument('-unld_tr', '--unld_train_file', type = str, default = None, help = 'unlabeled train file (for language model)')
    parser.add_argument('-unld_val', '--unld_val_file', type = str, default = None, help = 'unlabeled validation file (for language model)')
    parser.add_argument('-epochs', '--n_epochs', type = int, default = 50)
    parser.add_argument('-lstm_hd', '--lstm_hidden_dim', type = int, default = 256)
    parser.add_argument('-nfmaps', '--nfeature_maps', type = int, default = 200)
    parser.add_argument('-dense_hd', '--dense_hidden_dim', type = int, default = 256)
    parser.add_argument('-do', '--dropout', type = float, default = 0.5)
    parser.add_argument('-bsz', '--batch_size', type = int, default = 256)

    args = vars(parser.parse_args())

    return args


if __name__ == '__main__':

    main(parse_arguments())

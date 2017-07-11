# from keras.preprocessing.text import Tokenizer
# from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dropout
from keras.layers import Input, Embedding, merge, LSTM, Dense, Bidirectional, Lambda
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import optimizers
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
# from keras import backend as K
from TweetReader2 import TweetCorpus
from sklearn.metrics import classification_report

import numpy as np
import argparse
import os
import sys
import math

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def build_lm_model(kwargs):

    embedding_layer = Embedding(kwargs['nchars'], kwargs['hidden_size'], input_length = kwargs['max_seq_len'] - 1, mask_zero = True, trainable = True, name = 'embedding_layer1')

    # lstm1 = LSTM(kwargs['hidden_size'], dropout = kwargs['dropout'], recurrent_dropout = kwargs['dropout'], return_sequences = True, name = 'lstm1')

    lstm1 = LSTM(kwargs['hidden_size'], return_sequences = True, name = 'lstm1')

    # lstm2 = LSTM(kwargs['hidden_size'], dropout = kwargs['dropout'], recurrent_dropout = kwargs['dropout'], return_sequences = True, name = 'lstm2')

    lstm2 = LSTM(kwargs['hidden_size'], return_sequences = True, name = 'lstm2')

    sequence_input = Input(shape = (kwargs['max_seq_len'] - 1,), dtype = 'int32')

    embedded_sequences = embedding_layer(sequence_input)

    embedded_sequences = Dropout(kwargs['dropout'])(embedded_sequences)

    lstm1_op = lstm1(embedded_sequences)

    lstm1_op = Dropout(kwargs['dropout'])(lstm1_op)

    lstm2_op = lstm2(lstm1_op)

    lstm2_op = Dropout(kwargs['dropout'])(lstm2_op)

    lm_op = Dense(kwargs['nchars'], activation = 'softmax', name = 'lm_op')(lstm2_op)

    lm = Model(inputs = [sequence_input], outputs = lm_op)

    return lm

def build_clf_model(kwargs):

    embedding_layer = Embedding(kwargs['nchars'], kwargs['hidden_size'], input_length = kwargs['max_seq_len'], mask_zero = True, trainable = True, name = 'embedding_layer2')

    # lstm1 = LSTM(kwargs['hidden_size'], dropout = kwargs['dropout'], recurrent_dropout = kwargs['dropout'], return_sequences = True, name = 'lstm1')

    lstm1 = LSTM(kwargs['hidden_size'], return_sequences = True, name = 'lstm1')

    # lstm2 = LSTM(kwargs['hidden_size'], dropout = kwargs['dropout'], recurrent_dropout = kwargs['dropout'], return_sequences = True, name = 'lstm2')

    lstm2 = LSTM(kwargs['hidden_size'], return_sequences = True, name = 'lstm2')

    sequence_input = Input(shape = (kwargs['max_seq_len'],), dtype = 'int32')

    embedded_sequences = embedding_layer(sequence_input)

    embedded_sequences = Dropout(kwargs['dropout'])(embedded_sequences)

    lstm1_op = lstm1(embedded_sequences)

    lstm1_op = Dropout(kwargs['dropout'])(lstm1_op)

    lstm2_op = lstm2(lstm1_op)

    lstm2_op = Dropout(kwargs['dropout'])(lstm2_op)

    lstm2_op = Lambda(lambda x: x[:, -1, :])(lstm2_op)

    clf_op = Dense(kwargs['nclasses'], activation = 'softmax', name = 'clf_op')(lstm2_op)

    clf = Model(inputs = [sequence_input], outputs = clf_op)

    return clf

def train_lm_streaming(model, corpus, args):

    opt = optimizers.Nadam()

    model.compile(loss = 'categorical_crossentropy',
            optimizer = opt)

    model.summary()

    early_stopping = EarlyStopping(monitor = 'val_loss', patience = 3)

    bst_model_path = os.path.join(args['model_save_dir'], 'language_model.h5')

    model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only = True, save_weights_only = True)

    model.fit_generator(corpus.unld_tr_data.get_mini_batch(args['batch_size']),
                        int(math.floor(corpus.unld_tr_data.wc / args['batch_size'])),
                        epochs = 15,
                        verbose = 1,
                        callbacks = [early_stopping, model_checkpoint],
                        validation_data = corpus.unld_val_data.get_mini_batch(args['batch_size']),
                        validation_steps = int(math.floor(corpus.unld_val_data.wc / args['batch_size'])))

def train_lm(model, corpus, args):

    X_train, X_val, y_train, y_val = corpus.get_data_for_lm()

    opt = optimizers.Nadam()

    model.compile(loss = 'categorical_crossentropy',
            optimizer = opt)

    model.summary()

    early_stopping = EarlyStopping(monitor = 'val_loss', patience = 3)

    bst_model_path = os.path.join(args['model_save_dir'], 'language_model.h5')

    model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only = True, save_weights_only = True)

    hist = model.fit([X_train], y_train, \
            validation_data = ([X_val], y_val), \
            epochs = 15, batch_size = args['batch_size'], shuffle = True, \
            callbacks = [early_stopping, model_checkpoint])

def train_classifier(model, corpus, args):

    X_train, X_val, X_test, y_train, y_val, y_test = corpus.get_data_for_classification()

#     y_train = np_utils.to_categorical(y_train, num_classes = len(corpus.label2idx))
#
#     y_val = np_utils.to_categorical(y_val, num_classes = len(corpus.label2idx))
#
#     y_test = np_utils.to_categorical(y_test, num_classes = len(corpus.label2idx))

    opt = optimizers.Nadam()

    model.compile(loss = 'categorical_crossentropy',
            optimizer = opt,
            metrics = ['acc'])

    model.summary()

    early_stopping = EarlyStopping(monitor = 'val_acc', patience = 5)

    bst_model_path = os.path.join(args['model_save_dir'], 'classifier_model.h5')

    model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only = True, save_weights_only = True)

    hist = model.fit([X_train], y_train, \
            validation_data = ([X_val], y_val), \
            epochs = args['n_epochs'], batch_size = args['batch_size'], shuffle = True, \
            callbacks = [early_stopping, model_checkpoint])

    model.load_weights(bst_model_path)

    bst_val_score = min(hist.history['val_loss'])

    preds = model.predict([X_test], batch_size = args['batch_size'], verbose = 1)

    print(classification_report(np.argmax(y_test, axis = 1), np.argmax(preds, axis = 1), target_names = corpus.get_class_names()))

def generate_samples(model, corpus, num_samples):

    start_chars = ['f', 'o', 'rt ', '@', 'i', '_', 'n', 'g']
    # start_chars = ['G', 'E', 'R', 'C', 'H', 'I']
    # all_samples = []

    print '##############################################'

    for _ in xrange(num_samples):

        start_char = start_chars[np.random.randint(0, len(start_chars) - 1)]
        start_char_idx = corpus.char2idx[start_char]
        curr_sample = [start_char_idx]

        for _ in xrange(np.random.randint(50, 140)):
            seq_in = np.asarray([0 for _ in xrange(args['max_seq_len'] - 1 - len(curr_sample))] + curr_sample)
            inp = np.asarray([seq_in])
            prediction = model.predict(inp, verbose = 0)
            index = np.argmax(prediction[0][-1])
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

def main(args):

    # corpus = TweetCorpus(unlabeled_tweets_file = args['tweets_file'])
    # corpus = TweetCorpus(args['train_file'], args['val_file'], args['test_file'], args['tweets_file'])
    corpus = TweetCorpus(args['train_file'], args['val_file'], args['test_file'], args['unld_train_file'], args['unld_val_file'], args['dictionaries_file'])

    args['max_seq_len'] = corpus.max_len
    print 'max_seq_len: ', args['max_seq_len']
    args['nclasses'] = len(corpus.label2idx)
    print 'nclasses: ', args['nclasses']
    args['nchars'] = len(corpus.char2idx) + 1
    print 'nchars: ', args['nchars']

    if args['mode'] == 'lm':
        print 'Creating language model . . .'
        lm = build_lm_model(args)
        print 'Training language model . . .'
        # train_lm(lm, corpus, args)
        train_lm_streaming(lm, corpus, args)
        if os.path.isfile(os.path.join(args['model_save_dir'], 'language_model.h5')):
            print 'Loading weights from trained language model for text generation . . .'
            lm.load_weights(os.path.join(args['model_save_dir'], 'language_model.h5'), by_name = True)
        else:
            print 'No trained language model available . . .!'
            sys.exit(0)

        print 'Generating some fresh text . . .'
        generate_samples(lm, corpus, 20)

    elif args['mode'] == 'clf':
        print 'Creating classifier model . . .'
        clf = build_clf_model(args)
        # if the weights from the lm exists then use those weights instead
        if os.path.isfile(os.path.join(args['model_save_dir'], 'language_model.h5')):
            print 'Loading weights from trained language model . . .'
            clf.load_weights(os.path.join(args['model_save_dir'], 'language_model.h5'), by_name = True)
        print 'Training classifier model . . .'
        train_classifier(clf, corpus, args)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = '')

    parser.add_argument('train_file', type = str)
    parser.add_argument('val_file', type = str)
    parser.add_argument('test_file', type = str)
    parser.add_argument('unld_train_file', type = str)
    parser.add_argument('unld_val_file', type = str)
    parser.add_argument('dictionaries_file', type = str)
    parser.add_argument('model_save_dir', type = str)
    parser.add_argument('mode', type = str)
    parser.add_argument('--n_epochs', type = int, default = 50)
    parser.add_argument('--hidden_size', type = int, default = 256)
    parser.add_argument('--dropout', type = float, default = 0.5)
    parser.add_argument('--learning_rate', type = float, default = 0.01)

    parser.add_argument('--batch_size', type = int, default = 64)

    args = vars(parser.parse_args())

    main(args)

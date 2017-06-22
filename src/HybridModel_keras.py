# from keras.preprocessing.text import Tokenizer
# from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dropout
from keras.layers import Input, Embedding, merge, LSTM, Dense, Bidirectional, Lambda
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from keras import backend as K
from TweetReader import TweetCorpus
from sklearn.metrics import classification_report

import numpy as np
import argparse
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def reshape(x):
    pass


def build_lm_model(kwargs):

    embedding_layer = Embedding(kwargs['nchars'], kwargs['hidden_size'], input_length = kwargs['max_seq_len'] - 1, mask_zero = True, trainable = True, name = 'embedding_layer')

    lstm1 = LSTM(kwargs['hidden_size'], dropout = kwargs['dropout'], recurrent_dropout = kwargs['dropout'], return_sequences = True, name = 'lstm1')

    lstm2 = LSTM(kwargs['hidden_size'], dropout = kwargs['dropout'], recurrent_dropout = kwargs['dropout'], return_sequences = True, name = 'lstm2')

    sequence_input = Input(shape = (kwargs['max_seq_len'] - 1,), dtype = 'int32')

    embedded_sequences = embedding_layer(sequence_input)

    lstm1_op = lstm1(embedded_sequences)

    lstm1_op = Dropout(kwargs['dropout'])(lstm1_op)

    # lstm1_op = BatchNormalization()(lstm1_op)

    lstm2_op = lstm2(lstm1_op)

    lstm2_op = Dropout(kwargs['dropout'])(lstm2_op)

    # lstm2_op = Lambda(lambda x: x[:, -1, :])(lstm2_op)

    # lstm2_op = BatchNormalization()(lstm2_op)

    lm_op = Dense(kwargs['nchars'], activation = 'softmax', name = 'lm_op')(lstm2_op)

    lm = Model(inputs = [sequence_input], outputs = lm_op)

    return lm

def build_clf_model(kwargs):

    embedding_layer = Embedding(kwargs['nchars'], kwargs['hidden_size'], input_length = kwargs['max_seq_len'], mask_zero = True, trainable = True, name = 'embedding_layer')

    lstm1 = LSTM(kwargs['hidden_size'], dropout = kwargs['dropout'], recurrent_dropout = kwargs['dropout'], return_sequences = True, name = 'lstm1')

    lstm2 = LSTM(kwargs['hidden_size'], dropout = kwargs['dropout'], recurrent_dropout = kwargs['dropout'], return_sequences = True, name = 'lstm2')

    sequence_input = Input(shape = (kwargs['max_seq_len'],), dtype = 'int32')

    embedded_sequences = embedding_layer(sequence_input)

    lstm1_op = lstm1(embedded_sequences)

    lstm1_op = Dropout(kwargs['dropout'])(lstm1_op)

    # lstm1_op = BatchNormalization()(lstm1_op)

    lstm2_op = lstm2(lstm1_op)

    lstm2_op = Dropout(kwargs['dropout'])(lstm2_op)

    lstm2_op = Lambda(lambda x: x[:, -1, :])(lstm2_op)

    # lstm2_op = BatchNormalization()(lstm2_op)

    clf_op = Dense(kwargs['nclasses'], activation = 'softmax', name = 'clf_op')(lstm2_op)

    clf = Model(inputs = [sequence_input], outputs = clf_op)

    return clf

def get_one_hot_encoding(index, nclasses):

    ohvector = np.zeros(nclasses)
    ohvector[index] = 1

    return ohvector

def get_splits(X):

    y = []

    for x in X:
        curr_y = x[1:]
        _y = []
        for idx in curr_y:
            _y.append(get_one_hot_encoding(idx, args['nchars']))
        y.append(np.asarray(_y))

    y = np.asarray(y)
    X = X[:, :-1]

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.1, random_state = 42)

    return X_train, X_val, y_train, y_val

def train_lm(model, corpus, args):

    X = corpus.get_splits_for_lm()

    X_train, X_val, y_train, y_val = get_splits(X)

    model.compile(loss = 'categorical_crossentropy',
            optimizer = 'nadam')

    model.summary()

    early_stopping = EarlyStopping(monitor = 'val_loss', patience = 3)

    bst_model_path = os.path.join(args['model_save_dir'], 'language_model.h5')

    model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only = True, save_weights_only = True)

    hist = model.fit([X_train], y_train, \
            validation_data = ([X_val], y_val), \
            epochs = args['n_epochs'], batch_size = args['batch_size'], shuffle = True, \
            callbacks = [early_stopping, model_checkpoint])


def train_classifier(model, corpus, args):

    X_train, X_val, X_test, y_train, y_val, y_test = corpus.get_splits()

    y_train = np_utils.to_categorical(y_train, num_classes = len(corpus.label2idx))

    y_val = np_utils.to_categorical(y_val, num_classes = len(corpus.label2idx))

    y_test = np_utils.to_categorical(y_test, num_classes = len(corpus.label2idx))

    model.compile(loss = 'categorical_crossentropy',
            optimizer = 'nadam',
            metrics = ['acc'])

    model.summary()

    early_stopping = EarlyStopping(monitor = 'val_acc', patience = 3)

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

def get_emb_matrix(corpus, emb_dim = 300):

    emb_matrix = np.zeros((len(corpus.char2idx) + 1, emb_dim))

    for i in xrange(1, emb_matrix.shape[0]):
        emb_matrix[i] = np.random.uniform(-0.25, 0.25, emb_dim)

    return emb_matrix

def main(args):

    corpus = TweetCorpus(args['train_file'], args['val_file'], args['test_file'], args['tweets_file'])
    print 'len(vocab): ', len(corpus.char2idx)
    args['max_seq_len'] = corpus.max_len
    args['nclasses'] = len(corpus.label2idx)
    args['nchars'] = len(corpus.char2idx) + 1

    if args['mode'] == 'lm':
        print 'Creating language model . . .'
        lm = build_lm_model(args)
        print 'Training language model . . .'
        train_lm(lm, corpus, args)
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
    parser.add_argument('tweets_file', type = str)
    parser.add_argument('model_save_dir', type = str)
    parser.add_argument('mode', type = str)
    parser.add_argument('--n_epochs', type = int, default = 15)
    parser.add_argument('--hidden_size', type = int, default = 128)
    parser.add_argument('--dropout', type = float, default = 0.5)
    parser.add_argument('--learning_rate', type = float, default = 0.01)

    parser.add_argument('--batch_size', type = int, default = 32)

    args = vars(parser.parse_args())

    main(args)

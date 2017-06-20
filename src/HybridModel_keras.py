# from keras.preprocessing.text import Tokenizer
# from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dropout
from keras.layers import Input, Embedding, merge, LSTM, Dense, Bidirectional
from keras.models import Model
# from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import np_utils
from TweetReader import TweetCorpus
from sklearn.metrics import classification_report

import numpy as np
import argparse
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def build_models(emb_matrix, kwargs):

    nchars = emb_matrix.shape[0]

    emb_dim = emb_matrix.shape[1]

    # embedding_layer = Embedding(nchars, emb_dim, weights = [emb_matrix], input_length = kwargs['max_seq_len'], trainable = True)

    embedding_layer = Embedding(nchars, emb_dim, input_length = kwargs['max_seq_len'], mask_zero = True, trainable = True)

    lstm1 = LSTM(kwargs['hidden_size'], dropout = kwargs['dropout'], recurrent_dropout = kwargs['dropout'], return_sequences = True)
    # lstm1 = LSTM(kwargs['hidden_size'], dropout = kwargs['dropout'], recurrent_dropout = kwargs['dropout'])

    # lstm2 = LSTM(kwargs['hidden_size'], dropout = kwargs['dropout'], recurrent_dropout = kwargs['dropout'], return_sequences = True)
    lstm2 = LSTM(kwargs['hidden_size'], dropout = kwargs['dropout'], recurrent_dropout = kwargs['dropout'])

    sequence_input = Input(shape = (kwargs['max_seq_len'],), dtype = 'int32')

    embedded_sequences = embedding_layer(sequence_input)

    lstm1_op = lstm1(embedded_sequences)

    lstm1_op = Dropout(kwargs['dropout'])(lstm1_op)

    # lstm1_op = BatchNormalization()(lstm1_op)

    lstm2_op = lstm2(lstm1_op)

    lstm2_op = Dropout(kwargs['dropout'])(lstm2_op)

    # lstm2_op = BatchNormalization()(lstm2_op)

    # TODO: have to figure out how to reshape the output at this point

    lm_op = Dense(nchars, activation = 'softmax')(lstm2_op)

    clf_op = Dense(kwargs['nclasses'], activation = 'softmax')(lstm2_op)

    lm = Model(inputs = [sequence_input], outputs = lm_op)

    clf = Model(inputs = [sequence_input], outputs = clf_op)

    return lm, clf

def train_lm(model, corpus, model_save_dir, n_epochs, hidden_size, n_layers, dropout, learning_rate, batch_size):

    X_train = corpus.get_splits_for_lm()

    y_train = []

    for x in X_train:
        y_train.append(x[1:])

    y_train = np.asarray(y_train)
    X_train = X_train[:, :-1]

    model.compile(loss = 'categorical_crossentropy',
                  optimizer = 'nadam')

    model.summary()

    # early_stopping = EarlyStopping(monitor = 'val_acc', patience = 3)

    bst_model_path = 'language_model.h5'

    model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only = True, save_weights_only = True)

    hist = model.fit([X_train], y_train,
                     epochs = 25, batch_size = 200, shuffle = True,
                     callbacks = [model_checkpoint])

def train_classifier(model, corpus, args):

    X_train, X_val, X_test, y_train, y_val, y_test = corpus.get_splits()

    y_train = np_utils.to_categorical(y_train, num_classes = len(corpus.label2idx))

    y_val = np_utils.to_categorical(y_val, num_classes = len(corpus.label2idx))

    y_test = np_utils.to_categorical(y_test, num_classes = len(corpus.label2idx))

    model.compile(loss = 'categorical_crossentropy',
            optimizer = 'nadam',
            metrics = ['acc'])

    # model.summary()

    early_stopping = EarlyStopping(monitor = 'val_acc', patience = 5)

    bst_model_path = 'classifier' + '.h5'

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

    emb_matrix = get_emb_matrix(corpus, emb_dim = args['hidden_size'])

    args['max_seq_len'] = corpus.max_len
    args['nclasses'] = len(corpus.label2idx)

    lm, clf = build_models(emb_matrix, args)

    train_classifier(clf, corpus, args)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = '')

    parser.add_argument('train_file', type = str)
    parser.add_argument('val_file', type = str)
    parser.add_argument('test_file', type = str)
    parser.add_argument('tweets_file', type = str)
    parser.add_argument('model_save_dir', type = str)
    parser.add_argument('--n_epochs', type = int, default = 15)
    parser.add_argument('--hidden_size', type = int, default = 128)
    parser.add_argument('--dropout', type = float, default = 0.5)
    parser.add_argument('--learning_rate', type = float, default = 0.01)
    parser.add_argument('--batch_size', type = int, default = 32)

    args = vars(parser.parse_args())

    main(args)

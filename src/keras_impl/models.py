from keras.layers import Dropout
from keras.layers import Input, Embedding, LSTM, Dense, Lambda, RepeatVector
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import MaxPooling1D
from keras.layers.wrappers import TimeDistributed
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import optimizers

import os, math, numpy as np

class LanguageModel(object):

    '''
    Wrapper class around Keras Model API
    '''

    def __init__(self, W, kwargs):

        self.embedding_layer = Embedding(kwargs['nchars'], len(W[0]), input_length = kwargs['max_seq_len'] - 1, weights = [W], mask_zero = True, trainable = kwargs['trainable'], name = 'embedding_layer1')

        # lstm1 = LSTM(kwargs['lstm_hidden_dim'], dropout = kwargs['dropout'], recurrent_dropout = kwargs['dropout'], return_sequences = True, name = 'lstm1')

        self.lstm1 = LSTM(kwargs['lstm_hidden_dim'], return_sequences = True, name = 'lstm1')

        # lstm2 = LSTM(kwargs['lstm_hidden_dim'], dropout = kwargs['dropout'], recurrent_dropout = kwargs['dropout'], return_sequences = True, name = 'lstm2')

        self.lstm2 = LSTM(kwargs['lstm_hidden_dim'], return_sequences = True, name = 'lstm2')

        self.dense1 = Dense(kwargs['dense_hidden_dim'], activation = 'relu', name = 'dense1')

        self.lm_op_layer = Dense(kwargs['nchars'], activation = 'softmax', name = 'lm_op_layer')

        sequence_input = Input(shape = (kwargs['max_seq_len'] - 1,), dtype = 'int32')

        embedded_sequences = self.embedding_layer(sequence_input)

        embedded_sequences = Dropout(kwargs['dropout'])(embedded_sequences)

        lstm1_op = self.lstm1(embedded_sequences)

        lstm1_op = Dropout(kwargs['dropout'])(lstm1_op)

        lstm2_op = self.lstm2(lstm1_op)

        lstm2_op = Dropout(kwargs['dropout'])(lstm2_op)

        dense_op = self.dense1(lstm2_op)

        lm_op = self.lm_op_layer(dense_op)

        self.model = Model(inputs = [sequence_input], outputs = lm_op)

    def fit(self, corpus, args):

        opt = optimizers.Nadam()

        self.model.compile(loss = 'categorical_crossentropy',
                optimizer = opt)

        self.model.summary()

        early_stopping = EarlyStopping(monitor = 'val_loss', patience = 3)

        bst_model_path = os.path.join(args['model_save_dir'], 'language_model.h5')

        model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only = True, save_weights_only = True)

        self.model.fit_generator(corpus.unld_tr_data.get_mini_batch(args['batch_size']),
                            int(math.floor(corpus.unld_tr_data.wc / args['batch_size'])),
                            epochs = 20,
                            verbose = 2,
                            callbacks = [early_stopping, model_checkpoint],
                            validation_data = corpus.unld_val_data.get_mini_batch(args['batch_size']),
                            validation_steps = int(math.floor(corpus.unld_val_data.wc / args['batch_size'])))

class Classifier(object):

    def __init__(self, W, kwargs):

        self.embedding_layer = Embedding(kwargs['nchars'], len(W[0]), input_length = kwargs['max_seq_len'], weights = [W], mask_zero = True, trainable = kwargs['trainable'], name = 'embedding_layer2')

        # lstm1 = LSTM(kwargs['lstm_hidden_dim'], dropout = kwargs['dropout'], recurrent_dropout = kwargs['dropout'], return_sequences = True, name = 'lstm1')

        self.lstm1 = LSTM(kwargs['lstm_hidden_dim'], return_sequences = True, name = 'lstm1')

        # lstm2 = LSTM(kwargs['lstm_hidden_dim'], dropout = kwargs['dropout'], recurrent_dropout = kwargs['dropout'], return_sequences = True, name = 'lstm2')

        self.lstm2 = LSTM(kwargs['lstm_hidden_dim'], return_sequences = True, name = 'lstm2')

        self.dense1 = Dense(kwargs['dense_hidden_dim'], activation = 'relu', name = 'dense1')

        self.clf_op_layer = Dense(kwargs['nclasses'], activation = 'softmax', name = 'clf_op_layer')

        sequence_input = Input(shape = (kwargs['max_seq_len'],), dtype = 'int32')

        embedded_sequences = self.embedding_layer(sequence_input)

        embedded_sequences = Dropout(kwargs['dropout'])(embedded_sequences)

        lstm1_op = self.lstm1(embedded_sequences)

        lstm1_op = Dropout(kwargs['dropout'])(lstm1_op)

        lstm2_op = self.lstm2(lstm1_op)

        lstm2_op = Dropout(kwargs['dropout'])(lstm2_op)

        lstm2_op = Lambda(lambda x: x[:, -1, :])(lstm2_op)

        dense_op = self.dense1(lstm2_op)

        clf_op = self.clf_op_layer(dense_op)

        self.model = Model(inputs = [sequence_input], outputs = clf_op)

    def fit(self, X_train, X_val, X_test, y_train, y_val, class_weights, args):

        opt = optimizers.Nadam()

        self.model.compile(loss = 'categorical_crossentropy',
                optimizer = opt,
                metrics = ['acc'])

        self.model.summary()

        early_stopping = EarlyStopping(monitor = 'val_acc', patience = 10)

        bst_model_path = os.path.join(args['model_save_dir'], 'classifier_model.h5')

        model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only = True, save_weights_only = True)

        hist = self.model.fit([X_train], y_train, \
                validation_data = ([X_val], y_val), \
                epochs = args['n_epochs'], verbose = 2, batch_size = args['batch_size'], shuffle = True, \
                callbacks = [early_stopping, model_checkpoint], class_weight = class_weights)

        self.model.load_weights(bst_model_path)

        bst_val_score = min(hist.history['val_loss'])

        preds = self.model.predict([X_test], batch_size = args['batch_size'], verbose = 2)

        return np.argmax(preds, axis = 1)

class AutoEncoder(object):

    def __init__(self, W, kwargs):

        self.embedding_layer = Embedding(kwargs['nchars'], len(W[0]), input_length = kwargs['max_seq_len'], weights = [W], mask_zero = True, trainable = kwargs['trainable'], name = 'embedding_layer3')

        self.lstm1 = LSTM(kwargs['lstm_hidden_dim'], return_sequences = True, name = 'lstm1')

        self.lstm2 = LSTM(kwargs['lstm_hidden_dim'], name = 'lstm2')

        self.lstm3 = LSTM(kwargs['lstm_hidden_dim'], return_sequences = True, name = 'lstm3')

        # self.lstm4 = LSTM(kwargs['lstm_hidden_dim'], return_sequences = True, name = 'lstm4')

        self.ae_op_layer = TimeDistributed(Dense(kwargs['nchars'], activation = 'softmax', name = 'ae_op_layer'))

        sequence_input = Input(shape = (kwargs['max_seq_len'],), dtype = 'int32')

        embedded_sequences = self.embedding_layer(sequence_input)

        embedded_sequences = Dropout(kwargs['dropout'])(embedded_sequences)

        lstm1_op = self.lstm1(embedded_sequences)

        lstm1_op = Dropout(kwargs['dropout'])(lstm1_op)

        lstm2_op = self.lstm2(lstm1_op)

        encoded = Dropout(kwargs['dropout'])(lstm2_op)

        decoded = RepeatVector(kwargs['max_seq_len'])(encoded)

        decoded = self.lstm3(decoded)

        decoded = Dropout(kwargs['dropout'])(decoded)

        # decoded = self.lstm4(decoded)

        # decoded = Dropout(kwargs['dropout'])(decoded)

        ae_op = self.ae_op_layer(decoded)

        self.model = Model(inputs = [sequence_input], outputs = ae_op)

    def fit(self, corpus, args):

        opt = optimizers.Nadam()

        self.model.compile(loss = 'categorical_crossentropy',
                optimizer = opt)

        self.model.summary()

        early_stopping = EarlyStopping(monitor = 'val_loss', patience = 5)

        bst_model_path = os.path.join(args['model_save_dir'], 'autoencoder_model.h5')

        model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only = True, save_weights_only = True)

        self.model.fit_generator(corpus.unld_tr_data.get_mini_batch(args['batch_size']),
                            int(math.floor(corpus.unld_tr_data.wc / args['batch_size'])),
                            epochs = 20,
                            verbose = 2,
                            callbacks = [early_stopping, model_checkpoint],
                            validation_data = corpus.unld_val_data.get_mini_batch(args['batch_size']),
                            validation_steps = int(math.floor(corpus.unld_val_data.wc / args['batch_size'])))

class AutoEncoder_CNN(object):

    def __init__(self, W, kwargs):

        self.embedding_layer = Embedding(kwargs['nchars'], len(W[0]), input_length = kwargs['max_seq_len'], weights = [W], trainable = kwargs['trainable'], name = 'embedding_layer4')
        self.conv1 = Conv1D(kwargs['nfeature_maps'], 7, activation = 'relu')  # number of filters, filter width
        self.max_pool1 = MaxPooling1D(pool_size = 3)
        self.conv2 = Conv1D(kwargs['nfeature_maps'], 7, activation = 'relu')  # number of filters, filter width
        self.max_pool2 = MaxPooling1D(pool_size = 3)
        self.conv3 = Conv1D(kwargs['nfeature_maps'], 3, activation = 'relu')
        self.conv4 = Conv1D(kwargs['nfeature_maps'], 3, activation = 'relu')
        self.lstm1 = LSTM(kwargs['lstm_hidden_dim'], name = 'lstm1')  # part of encoder
        self.lstm2 = LSTM(kwargs['lstm_hidden_dim'], return_sequences = True, name = 'lstm2')  # part of decoder
        self.lstm3 = LSTM(kwargs['lstm_hidden_dim'], return_sequences = True, name = 'lstm3')  # part of decoder
        self.ae_op_layer = TimeDistributed(Dense(kwargs['nchars'], activation = 'softmax', name = 'ae_op_layer'))

        sequence_input = Input(shape = (kwargs['max_seq_len'],), dtype = 'int32')
        embedded_sequences = self.embedding_layer(sequence_input)
        # embedded_sequences = Dropout(kwargs['dropout'])(embedded_sequences)
        conv1_op = self.conv1(embedded_sequences)
        max_pool1_op = self.max_pool1(conv1_op)
        # max_pool1_op = Dropout(kwargs['dropout'])(max_pool1_op)
        conv2_op = self.conv2(max_pool1_op)
        max_pool2_op = self.max_pool2(conv2_op)
        # max_pool2_op = Dropout(kwargs['dropout'])(max_pool2_op)
        conv3_op = self.conv3(max_pool2_op)
        conv4_op = self.conv4(conv3_op)
        encoded = self.lstm1(conv4_op)
        # encoded = Dropout(kwargs['dropout'])(encoded)
        decoded = RepeatVector(kwargs['max_seq_len'])(encoded)
        decoded = self.lstm2(decoded)
        # decoded = Dropout(kwargs['dropout'])(decoded)
        decoded = self.lstm3(decoded)
        decoded = Dropout(kwargs['dropout'])(decoded)
        ae_op = self.ae_op_layer(decoded)
        self.model = Model(inputs = [sequence_input], outputs = ae_op)

    def fit(self, corpus, args):

        opt = optimizers.Nadam()

        self.model.compile(loss = 'categorical_crossentropy',
                optimizer = opt)

        self.model.summary()

        early_stopping = EarlyStopping(monitor = 'val_loss', patience = 5)

        bst_model_path = os.path.join(args['model_save_dir'], 'cnn_autoencoder_model.h5')

        model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only = True, save_weights_only = True)

        self.model.fit_generator(corpus.unld_tr_data.get_mini_batch(args['batch_size']),
                            int(math.floor(corpus.unld_tr_data.wc / args['batch_size'])),
                            epochs = 20,
                            verbose = 2,
                            callbacks = [early_stopping, model_checkpoint],
                            validation_data = corpus.unld_val_data.get_mini_batch(args['batch_size']),
                            validation_steps = int(math.floor(corpus.unld_val_data.wc / args['batch_size'])))

from keras.layers import Dropout
from keras.layers import Input, Embedding, LSTM, Dense, Lambda
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import optimizers

import os, math, numpy as np

class LanguageModel(object):

    '''
    Wrapper class around Keras Model API
    '''

    def __init__(self, kwargs):

        self.embedding_layer = Embedding(kwargs['nchars'], kwargs['emb_dim'], input_length = kwargs['max_seq_len'] - 1, mask_zero = True, trainable = True, name = 'embedding_layer1')

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

    def __init__(self, kwargs):

        self.embedding_layer = Embedding(kwargs['nchars'], kwargs['emb_dim'], input_length = kwargs['max_seq_len'], mask_zero = True, trainable = True, name = 'embedding_layer2')

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

from keras import optimizers
import keras
from keras.callbacks import Callback
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Conv1D
from keras.layers import Dropout
from keras.layers import Input, Embedding, LSTM, Dense, Lambda, RepeatVector
from keras.layers import MaxPooling1D, GlobalMaxPooling1D, concatenate
from keras.layers.wrappers import TimeDistributed
from keras.models import Model
import os, math, numpy as np
from sklearn.metrics import f1_score, accuracy_score


class MyCallback(Callback):
    """
    My custom callback
    """

    def __init__(self, val_data, filepath, min_delta = 0, patience = 0, verbose = 0,
                 save_best_only = False, save_weights_only = False):
        super(MyCallback, self).__init__()
        self.patience = patience
        self.min_delta = min_delta
        self.wait = 0
        self.stopped_epoch = 0
        self.verbose = verbose
        self.val_data = val_data
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.monitor_op = np.greater

    def on_train_begin(self, logs = None):
        # Allow instances to be re-used
        self.wait = 0
        self.stopped_epoch = 0
        self.best = -np.Inf

    def on_epoch_end(self, epoch, logs = None):

        y_pred = np.argmax(self.model.predict(self.val_data[0]), axis = 1)
        y_true = np.argmax(self.val_data[1], axis = 1)
        macro_f1 = f1_score(y_true, y_pred, average = 'macro')
        acc = accuracy_score(y_true, y_pred)
        print ('Val macro F1: %.4f, Val acc: %.4f' % (macro_f1, acc))
        # check if there is any improvement
        if self.monitor_op(macro_f1 - self.min_delta, self.best):
            self.best = macro_f1
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True

        filepath = self.filepath.format(epoch = epoch, **logs)
        if self.save_best_only:
            if self.wait == 0:
                if self.verbose > 0:
                    print('Saving model to %s' % (filepath))
                if self.save_weights_only:
                    self.model.save_weights(filepath, overwrite = True)
                else:
                    self.model.save(filepath, overwrite = True)
        else:
            if self.verbose > 0:
                print('Saving model to %s' % (filepath))
            if self.save_weights_only:
                self.model.save_weights(filepath, overwrite = True)
            else:
                self.model.save(filepath, overwrite = True)

    def on_train_end(self, logs = None):
        if self.stopped_epoch > 0 and self.verbose > 0:
            print('Early stopping . . .')


class LSTMLanguageModel(object):
    '''
    Wrapper class around Keras Model API
    '''

    def __init__(self, W, kwargs):

        self.embedding_layer = Embedding(kwargs['ntokens'], len(W[0]), input_length = kwargs['max_seq_len'] - 1, weights = [W], mask_zero = True, trainable = kwargs['trainable'], name = 'embedding_layer')

        # lstm1 = LSTM(kwargs['lstm_hidden_dim'], dropout = kwargs['dropout'], recurrent_dropout = kwargs['dropout'], return_sequences = True, name = 'lstm1')

        self.lstm1 = LSTM(kwargs['lstm_hidden_dim'], return_sequences = True, name = 'lstm1')

        # lstm2 = LSTM(kwargs['lstm_hidden_dim'], dropout = kwargs['dropout'], recurrent_dropout = kwargs['dropout'], return_sequences = True, name = 'lstm2')

        self.lstm2 = LSTM(kwargs['lstm_hidden_dim'], return_sequences = True, name = 'lstm2')

        # self.dense1 = Dense(kwargs['dense_hidden_dim'], activation = 'relu', name = 'dense1')

        self.dense1 = TimeDistributed(Dense(kwargs['dense_hidden_dim'], activation = 'relu', name = 'dense1'))

        # self.lm_op_layer = Dense(kwargs['ntokens'], activation = 'softmax', name = 'lm_op_layer')

        self.lm_op_layer = TimeDistributed(Dense(kwargs['ntokens'], activation = 'softmax', name = 'lm_op_layer'))

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

        X_t, X_v, y_t, y_v = corpus.get_data_for_lm()

        opt = optimizers.Nadam()

        self.model.compile(loss = keras.losses.sparse_categorical_crossentropy,
                optimizer = opt)

        self.model.summary()

        early_stopping = EarlyStopping(monitor = 'val_loss', patience = 3)

        bst_model_path = os.path.join(args['model_save_dir'], 'lstm_language_model.h5')

        model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only = True, save_weights_only = True)

#         self.model.fit_generator(corpus.unld_tr_data.get_mini_batch(args['batch_size']),
#                             int(math.floor(corpus.unld_tr_data.wc / args['batch_size'])),
#                             epochs = 20,
#                             verbose = 2,
#                             callbacks = [early_stopping, model_checkpoint],
#                             validation_data = corpus.unld_val_data.get_mini_batch(args['batch_size']),
#                             validation_steps = int(math.floor(corpus.unld_val_data.wc / args['batch_size'])))

        self.model.fit([X_t], y_t, \
                validation_data = ([X_v], y_v), \
                epochs = 15, verbose = 2, batch_size = args['batch_size'], shuffle = True, \
                callbacks = [early_stopping, model_checkpoint])


class LSTMClassifier(object):

    def __init__(self, W, kwargs):

        self.embedding_layer = Embedding(kwargs['ntokens'], len(W[0]), input_length = kwargs['max_seq_len'], weights = [W], mask_zero = True, trainable = kwargs['trainable'], name = 'embedding_layer')

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

        self._intermediate_layer_model = Model(inputs = self.model.input,
                                 outputs = self.model.get_layer('dense1').output)

    def fit(self, X_train, X_val, X_test, y_train, y_val, class_weights, args):

        opt = optimizers.Nadam()

        self.model.compile(loss = 'categorical_crossentropy',
                optimizer = opt,
                metrics = ['acc'])

        self.model.summary()

        # early_stopping = MyEarlyStopping(([X_val], y_val), patience = 5, verbose = 1)

        bst_model_path = os.path.join(args['model_save_dir'], 'lstm_classifier_model_' + args['ts'] + '.h5')

        # model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only = True, save_weights_only = True)

        callback = MyCallback(([X_val], y_val), bst_model_path, patience = 5, verbose = 1, save_best_only = True, save_weights_only = True)

        hist = self.model.fit([X_train], y_train, \
                validation_data = None, \
                epochs = args['n_epochs'], verbose = 2, batch_size = args['batch_size'], shuffle = True, \
                callbacks = [callback], class_weight = class_weights)

        return self.predict(X_test, bst_model_path, args['batch_size'])

    def predict(self, X_test, model_path, batch_size):
        self.model.load_weights(model_path)
        preds = self.model.predict([X_test], batch_size = batch_size, verbose = 2)
        representations = self._intermediate_layer_model.predict(X_test, batch_size = batch_size, verbose = 2)
        return preds, representations


class CNNClassifier(object):

    def __init__(self, W, kwargs):
        self.embedding_layer = Embedding(kwargs['ntokens'], len(W[0]), input_length = kwargs['max_seq_len'], weights = [W], mask_zero = False, trainable = kwargs['trainable'], name = 'embedding_layer')

        self.conv_ls = []
        for ksz in kwargs['kernel_sizes']:
            self.conv_ls.append(Conv1D(kwargs['nfeature_maps'], ksz, name = 'conv' + str(ksz)))

        self.mxp_l = GlobalMaxPooling1D()
        self.dense1 = Dense(kwargs['dense_hidden_dim'], activation = 'relu', name = 'dense1')

        self.clf_op_layer = Dense(kwargs['nclasses'], activation = 'softmax', name = 'clf_op_layer')

        sequence_input = Input(shape = (kwargs['max_seq_len'],), dtype = 'int32')

        embedded_sequences = self.embedding_layer(sequence_input)

        embedded_sequences = Dropout(kwargs['dropout'])(embedded_sequences)

        conv_mxp_ops = []
        for conv_l in self.conv_ls:
            _tmp_op = conv_l(embedded_sequences)
            _tmp_op = self.mxp_l(_tmp_op)
            _tmp_op = Dropout(kwargs['dropout'])(_tmp_op)
            conv_mxp_ops.append(_tmp_op)

        conv_op = concatenate(conv_mxp_ops, axis = 1)

        dense_op = self.dense1(conv_op)

        clf_op = self.clf_op_layer(dense_op)

        self.model = Model(inputs = [sequence_input], outputs = clf_op)

        self._intermediate_layer_model = Model(inputs = self.model.input,
                                 outputs = self.model.get_layer('dense1').output)

    def fit(self, X_train, X_val, X_test, y_train, y_val, class_weights, args):

        opt = optimizers.Nadam()

        self.model.compile(loss = 'categorical_crossentropy',
                optimizer = opt,
                metrics = ['acc'])

        self.model.summary()

        # early_stopping = MyEarlyStopping(([X_val], y_val), patience = 5, verbose = 1)

        bst_model_path = os.path.join(args['model_save_dir'], 'cnn_classifier_model_' + args['ts'] + '.h5')

        # model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only = True, save_weights_only = True)

        callback = MyCallback(([X_val], y_val), bst_model_path, patience = 5, verbose = 1, save_best_only = True, save_weights_only = True)

        hist = self.model.fit([X_train], y_train, \
                validation_data = None, \
                epochs = args['n_epochs'], verbose = 2, batch_size = args['batch_size'], shuffle = True, \
                callbacks = [callback], class_weight = class_weights)

        return self.predict(X_test, bst_model_path, args['batch_size'])

    def predict(self, X_test, model_path, batch_size):
        self.model.load_weights(model_path)
        preds = self.model.predict([X_test], batch_size = batch_size, verbose = 2)
        representations = self._intermediate_layer_model.predict(X_test, batch_size = batch_size, verbose = 2)
        return preds, representations


class AutoEncoder(object):

    def __init__(self, W, kwargs):

        self.embedding_layer = Embedding(kwargs['ntokens'], len(W[0]), input_length = kwargs['max_seq_len'], weights = [W], mask_zero = True, trainable = kwargs['trainable'], name = 'embedding_layer')

        self.lstm1 = LSTM(kwargs['lstm_hidden_dim'], return_sequences = True, name = 'lstm1')

        self.lstm2 = LSTM(kwargs['lstm_hidden_dim'], name = 'lstm2')

        self.lstm3 = LSTM(kwargs['lstm_hidden_dim'], return_sequences = True, name = 'lstm3')

        # self.lstm4 = LSTM(kwargs['lstm_hidden_dim'], return_sequences = True, name = 'lstm4')

        self.ae_op_layer = TimeDistributed(Dense(kwargs['ntokens'], activation = 'softmax', name = 'ae_op_layer'))

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

        self.embedding_layer = Embedding(kwargs['ntokens'], len(W[0]), input_length = kwargs['max_seq_len'], weights = [W], trainable = kwargs['trainable'], name = 'embedding_layer')
        self.conv1 = Conv1D(kwargs['nfeature_maps'], 7, activation = 'relu')  # number of filters, filter width
        self.max_pool1 = MaxPooling1D(pool_size = 3)
        self.conv2 = Conv1D(kwargs['nfeature_maps'], 7, activation = 'relu')  # number of filters, filter width
        self.max_pool2 = MaxPooling1D(pool_size = 3)
        self.conv3 = Conv1D(kwargs['nfeature_maps'], 3, activation = 'relu')
        self.conv4 = Conv1D(kwargs['nfeature_maps'], 3, activation = 'relu')
        self.lstm1 = LSTM(kwargs['lstm_hidden_dim'], name = 'lstm1')  # part of encoder
        self.lstm2 = LSTM(kwargs['lstm_hidden_dim'], return_sequences = True, name = 'lstm2')  # part of decoder
        self.lstm3 = LSTM(kwargs['lstm_hidden_dim'], return_sequences = True, name = 'lstm3')  # part of decoder
        self.ae_op_layer = TimeDistributed(Dense(kwargs['ntokens'], activation = 'softmax', name = 'ae_op_layer'))

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

        opt = optimizers.Adam(clipvalue = 5.0)

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

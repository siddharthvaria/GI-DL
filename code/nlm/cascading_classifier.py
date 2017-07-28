import keras
from keras.models import Model, load_model
from keras.layers import Input, LSTM, Bidirectional, Dense, RepeatVector, BatchNormalization, concatenate, Embedding, Dropout

import numpy as np

class Classifier():
    def __init__(self, timesteps=140, latent_dim=128, dim=2, bidirectional=True,
                 embedding=False):
        # normally dimension should be 1, but added blank extra dimension bc of
        # theano weirdness
        inputs = Input((timesteps, dim))
        back_inputs = Input((timesteps, dim))

        self.bidirectional = bidirectional
        
        encoder = []
        fe = []
        be = []

        # encoder
        if embedding:
            encoder = Embedding(100, 64, input_length=timesteps)(inputs)
            encoder = Dropout(0.5)(encoder)
            encoder = LSTM(latent_dim, return_sequences=True)(encoder)
        else:
            encoder = LSTM(latent_dim, return_sequences=True)(inputs)
#        encoder = BatchNormalization()(encoder)
        encoder = Dropout(0.5)(encoder)
#        encoder = LSTM(latent_dim, return_sequences=True)(encoder)
        encoder = LSTM(latent_dim)(encoder)
#        encoder = Dropout(0.5)(encoder)

        # bidirectional encoder
        if embedding:
            fe = Embedding(100, 64, input_length=timesteps)(inputs)
#            fe = Dropout(0.5)(fe)
            fe = LSTM(latent_dim, return_sequences=True)(fe)
        else:
            fe = LSTM(latent_dim, return_sequences=True)(inputs)
#        fe = BatchNormalization()(fe)
#        fe = Dropout(0.5)(fe)
#        fe = LSTM(latent_dim, return_sequences=True)(fe)
#        fe = Dropout(0.5)
        fe = LSTM(latent_dim)(fe)

        if embedding:
            be = Embedding(100, 64, input_length=timesteps)(back_inputs)
#            be = Dropout(0.5)(be)
            be = LSTM(latent_dim, return_sequences=True)(be)
        else:
            be = LSTM(latent_dim, return_sequences=True)(back_inputs)
#        be = BatchNormalization()(be)
#        be = Dropout(0.5)(be)
#        be = LSTM(latent_dim, return_sequences=True)(be)
#        be = Dropout(0.5)(be)
        be = LSTM(latent_dim)(be)

        # decoder
        decoder = None

        if bidirectional:
            decoder = concatenate([fe, be])
            decoder = RepeatVector(timesteps)(decoder)
        else:
            decoder = RepeatVector(timesteps)(encoder)
        decoder = LSTM(latent_dim, return_sequences=True)(decoder)
        decoder = LSTM(dim, return_sequences=True)(decoder)
        
        # loss/aggress or none classifier (emotion or none)
        classify_em = Dense(latent_dim)(encoder)
#        classify = concatenate([fe, be])
#        classify_em = Dense(64)(classify_em)
#        classify_em = Dense(64)(classify_em)
        classify_em = Dense(2, activation='softmax')(classify_em)

        # loss or aggress classifier
        classify_la = Dense(latent_dim)(encoder)
#        classify_la = Dense(64)(classify_la)
#        classify_la = Dense(64)(classify_la)
        classify_la = Dense(2, activation='softmax')(classify_la)

        # models
        if bidirectional:
            self.autoencoder = Model(inputs=[inputs, back_inputs], outputs=[decoder])            
            self.em_classifier = Model(inputs=[inputs, back_inputs], outputs=[classify_em])
            self.la_classifier = Model(inputs=[inputs, back_inputs], outputs=[classify_la])

        else:
            self.autoencoder = Model(inputs, outputs=[decoder])
            self.em_classifier = Model(inputs, outputs=[classify_em])
            self.la_classifier = Model(inputs, outputs=[classify_la])

        self.autoencoder.compile(optimizer='adam', loss='categorical_crossentropy')
        self.em_classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        self.la_classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    def fit_auto(self, X, e=1, b=1):
        if self.bidirectional:
            out = X[0]
        else:
            out = X

        self.autoencoder.fit(X, out, epochs=e, batch_size=b)
        
    def predict_auto(self, X):
        return self.autoencoder.predict(X)
        
    def fit_classif(self, X, Y, validation_data=None, e=16, b=128,
                    class_weight={0:1., 1:1., 2:1.}):
        if self.bidirectional:
            em_x_f, em_y, la_x_f, la_y = cascade(X[0], Y)
            em_x_b, em_y, la_x_b, la_y = cascade(X[1], Y)

#            print '[fit-classif] ' + str(np.shape(em_x_f))
#            print '[fit-classif] ' + str(np.shape(em_x_b))

            self.em_classifier.fit([em_x_f, em_x_b], em_y, epochs=e,
                                   batch_size=b, class_weight=class_weight)#, callbacks=[stats])
            self.la_classifier.fit([la_x_f, la_x_b], la_y, epochs=e,
                                   batch_size=b, class_weight=class_weight)
        else:
            em_x, em_y, la_x, la_y = cascade(X, Y)
            self.em_classifier.fit(em_x, em_y, epochs=e, batch_size=b,
                                   class_weight=class_weight)
            self.la_classifier.fit(la_x, la_y, epochs=e, batch_size=b,
                                   class_weight=class_weight)

    def predict(self, X):
        em_preds = self.em_classifier.predict(X)
        la_preds = self.la_classifier.predict(X)

        preds = []

#        print '[predict] ' + str(np.shape(X))

        l = len(X[0]) if self.bidirectional else len(X)

        for i in range(l):
            if em_preds[i][0] > em_preds[i][1]:
                preds.append([0, 1, 0])
            elif la_preds[i][0] > la_preds[i][1]:
                preds.append([1, 0, 0])
            else:
                preds.append([0, 0, 1])

        preds = np.array(preds)

#        print '[predict] ' + str(np.shape(preds))

        for i in range(len(preds)):
            print '[predict] ' + str(em_preds[i]) + ' / ' + str(la_preds[i])

        return preds
        
    def evaluate(self, X, Y):
#        print self.classifier.predict(X)
        if self.bidirectional:
            em_x_f, em_y, la_x_f, la_y = cascade(X[0], Y)
            em_x_b, em_y, la_x_b, la_y = cascade(X[1], Y)

            return self.em_classifier.evaluate([em_x_f, em_x_b], em_y), self.la_classifier.evaluate([la_x_f, la_x_b], la_y)

        em_x, em_y, la_x, la_y = cascade(X, Y)

        return self.em_classifier.evaluate(em_x, em_y), self.la_classifier.evaluate(la_x, la_y)

def cascade(X, Y):
    em_x = []
    em_y = []
    la_x = []
    la_y = []
    
    for i in range(len(X)):
#        if Y[i][1] == 1:
#            print '[cascade] emotion - [1, 0]'

        em_x.append(X[i])
        em_y.append([1, 0] if Y[i][1] == 1 else [0, 1])
        
        if Y[i][1] == 0:
#            if Y[i][0] == 1:
#                print '[cascade] emotion - [0, 1]; loss/aggression - [1, 0]'
#            else:
#                print '[cascade] emotion - [0, 1]; loss/aggression - [0, 1]'

            la_x.append(X[i])
            la_y.append([1, 0] if Y[i][0] == 1 else [0, 1])

    em_x = np.array(em_x)
    em_y = np.array(em_y)
    la_x = np.array(la_x)
    la_y = np.array(la_y)

    return em_x, em_y, la_x, la_y

class Stats(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.scores = []

    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(logs.get('loss'))
        y_pred = self.model.predict(self.model.training_data[0])
        y_true = self.model.training_data[1]

        scores = [0, 0, 0]
        for i in range(3):
            scores[i] = fmeasure(y_pred, y_true, i)
            
        self.scores.append(scores)
    
def precision(ypred, ytrue, c):
    tp = 0.
    p = 0.

    for i in range(len(ypred)):
        if ypred[i][c] > ypred[i][c-1] and ypred[i][c] > ypred[i][c-2]:
            p += 1.

            if ytrue[i][c] == 1:
                tp += 1.

#    print '[p] ' + str(p)
#    print '[p] ' + str(tp)

    return 0 if p == 0 else tp/p

def recall(ypred, ytrue, c):
    tp = 0.
    t = 0.
        
    for i in range(len(ytrue)):
        if ytrue[i][c] == 1:
            t += 1.

            if ypred[i][c] > ypred[i][c-1] and ypred[i][c] > ypred[i][c-2]:
                tp += 1.

#    print '[r] ' + str(tp)
#    print '[r] ' + str(t)

    return None if t == 0 else tp/t

def fmeasure(ypred, ytrue, c):
    p = precision(ypred, ytrue, c)
    r = recall(ypred, ytrue, c)

    if p == None or r == None:
        return None

    if p == 0 and r == 0:
        return None

    return 2 * (p * r)/(p + r)

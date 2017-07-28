import keras
from keras.models import Model, load_model
from keras.layers import Input, LSTM, Bidirectional, Dense, RepeatVector, BatchNormalization, concatenate, Embedding, Dropout, Activation, TimeDistributed, RepeatVector, Lambda
from keras import backend as K
from keras import metrics

import numpy as np
import pickle

from keras_vae_example import sampling, CustomVariationalLayer

batch_size = 140
original_dim = 2
latent_dim = 2
intermediate_dim = 140
epochs = 8
epsilon_std = 1.0

class Classifier():
    def __init__(self, timesteps=140, dim=2, embedding=False, bidirectional=False):
        print '[classifier-init] initializing layers...'

        # normally dimension should be 1, but added blank extra dimension bc of
        # theano weirdness
        x = Input((timesteps, dim))

        h = Dense(intermediate_dim, activation='relu')(x)
        z_mu = Dense(latent_dim)(h)
        z_log_sigma = Dense(latent_dim)(h)
        z = Lambda(sampling, output_shape=(latent_dim,))([z_mu, z_log_sigma])

        decoder_h = Dense(intermediate_dim, activation='relu')
        decoder_mu = Dense(original_dim, activation='sigmoid')
        h_decoded = decoder_h(z)
        x_decoded_mu = decoder_mu(h_decoded)
        
        y = CustomVariationalLayer()([x, x_decoded_mu])

        decoder = Dense(intermediate_dim)(y)
#        decoder = RepeatVector(timesteps)(x_decoded_mu)
        decoder = LSTM(intermediate_dim, return_sequences=True)(decoder)
        decoder = LSTM(dim, return_sequences=True)(decoder)
        
        # classifier
        classify = Dense(intermediate_dim)(z)
#        classify = concatenate([fe, be])
        classify = Dense(intermediate_dim)(classify)
        classify = Dense(intermediate_dim)(classify)
        classify = Dense(3, activation='softmax')(classify)

        print '[classifier-init] defining models...'

        # models
        self.autoencoder = Model(inputs=[x], outputs=[y])
#        self.autoencoder = Model(inputs=[inputs, back_inputs], outputs=[decoder])
        self.classifier = Model(inputs=[x], outputs=[classify])
#        self.classifier = Model(inputs=[inputs, back_inputs], outputs=[classify])

        print '[classifier-init] compiling models...'

        self.autoencoder.compile(optimizer='adam', loss=None)
#        self.classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
        self.classifier.compile(optimizer='adam', loss='mse', metrics=['categorical_accuracy'])

        print 

    def fit_auto(self, X, e=1, b=1):
        self.autoencoder.fit(X, X, epochs=e, batch_size=b)
        
    def fit_classif(self, X, Y, validation_data=None, e=8, b=140):
        print '[fit-classif] ' + str(np.shape(X))
        print '[fit-classif] ' + str(np.shape(Y))
        self.classifier.fit(X, Y)
        return
        if validation_data:
            self.classifier.fit(X, Y, epochs=e, batch_size=b,
                                validation_set=validation_data)#, callbacks=[stats])
        else:
            self.classifier.fit(X, Y, epochs=e, batch_size=b)

    def predict(self, X):
        return self.classifier.predict(X)
        
    def evaluate(self, X, Y):
#        print self.classifier.predict(X)
        return self.classifier.evaluate(X, Y)

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
        if ypred[i][c] == 1:
            p += 1.

            if ytrue[i] == c:
                tp += 1.

    return None if p == 0 else tp/p

def recall(ypred, ytrue, c):
    tp = 0.
    t = 0.
        
    for i in range(len(ytrue)):
        if ytrue[i] == c:
            t += 1.

            if ypred[i][c] == 1:
                tp += 1.

    return None if t == 0 else tp/t

def fmeasure(ypred, ytrue, c):
    p = precision(ypred, ytrue, c)
    r = recall(ypred, ytrue, c)

    if p == None or r == None:
        return None

    if p == 0 and r == 0:
        return None

    return 2 * (p * r)/(p + r)

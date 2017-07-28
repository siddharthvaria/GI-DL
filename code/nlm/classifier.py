import keras
from keras.models import Model, load_model
from keras.layers import Input, LSTM, Bidirectional, Dense, RepeatVector, BatchNormalization, concatenate, Embedding, Dropout

class Classifier():
    def __init__(self, timesteps=140, latent_dim=128, dim=2, bidirectional=True,
                 embedding=False):
        # normally dimension should be 1, but added blank extra dimension bc of
        # theano weirdness
        inputs = Input((timesteps, dim))
        back_inputs = Input((timesteps, dim))

        self.bidirectional = bidirectional
        
        # encoder
        if embedding:
            encoder = Embedding(100, 64, input_length=timesteps)(inputs)
            encoder = Dropout(0.5)(encoder)
            encoder = LSTM(latent_dim, return_sequences=True)(encoder)
        else:
            encoder = LSTM(latent_dim, return_sequences=True)(inputs)
        encoder = Dropout(0.5)(encoder)
        encoder = LSTM(latent_dim)(encoder)
        encoder = Dropout(0.5)(encoder)

        # bidirectional encoder
        fe = LSTM(latent_dim, return_sequences=True)(inputs)
        fe = BatchNormalization()(fe)
        fe = LSTM(latent_dim, return_sequences=True)(fe)
        fe = LSTM(latent_dim)(fe)
        be = LSTM(latent_dim, return_sequences=True)(back_inputs)
        be = BatchNormalization()(be)
        be = LSTM(latent_dim, return_sequences=True)(be)
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
        
        # classifier
        classify = Dense(latent_dim)(encoder)
#        classify = concatenate([fe, be])
#        classify = Dense(latent_dim)(classify)
#        classify = Dense(latent_dim)(classify)
        classify = Dense(3, activation='softmax')(classify)

        # models
        if bidirectional:
            self.autoencoder = Model(inputs=[inputs, back_inputs], outputs=[decoder])            
            self.classifier = Model(inputs=[inputs, back_inputs], outputs=[classify])
        else:
            self.autoencoder = Model(inputs, outputs=[decoder])
            self.classifier = Model(inputs, outputs=[classify])


        self.autoencoder.compile(optimizer='adam', loss='categorical_crossentropy')
        self.classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

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
        self.classifier.fit(X, Y, epochs=e, batch_size=b,
                            class_weight=class_weight)#, callbacks=[stats])

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

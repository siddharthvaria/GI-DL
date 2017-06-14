import keras
from keras.models import Model, load_model
from keras.layers import Input, LSTM, Dense, RepeatVector, BatchNormalization

class Classifier():
    def __init__(self, timesteps, latent_dim=128, dim=2):
        # normally dimension should be 1, but added blank extra dimension bc of
        # theano weirdness
        inputs = Input((timesteps, dim))
        
        # encoder
        encoder = LSTM(latent_dim, return_sequences=True)(inputs)
        encoder = BatchNormalization()(encoder)
        encoder = LSTM(latent_dim, return_sequences=True)(encoder)
        encoder = LSTM(latent_dim)(encoder)

        # decoder
        decoder = RepeatVector(timesteps)(encoder)
#        decoder = LSTM(latent_dim)(encoder)
#        decoder = RepeatVector(timesteps)(decoder)
        decoder = LSTM(latent_dim, return_sequences=True)(decoder)
        decoder = LSTM(dim, return_sequences=True)(decoder)
        
        # classifier
        classify = Dense(latent_dim)(encoder)
        classify = Dense(latent_dim)(encoder)
        classify = Dense(3, activation='softmax')(classify)

        # models
        self.autoencoder = Model(inputs=[inputs], outputs=[decoder])
        self.classifier = Model(inputs=[inputs], outputs=[classify])

        self.autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
        self.classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    def fit_auto(self, X, e=1, b=1):
        self.autoencoder.fit(X, X, epochs=e, batch_size=b)
        
    def fit_classif(self, X, Y, validation_data=None, e=8, b=128):
        self.classifier.fit(X, Y, epochs=e, batch_size=b)

    def predict(self, X):
        return self.classifier.predict(X)
        
    def evaluate(self, X, Y):
#        print self.classifier.predict(X)
        return self.classifier.evaluate(X, Y)

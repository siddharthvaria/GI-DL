from keras.models import Model, load_model
from keras.layers import Input, LSTM, Dense, RepeatVector

# autoencoder/encoder + classifier for tweets
class Classifier:
    def __init__(self, timesteps, input_dim=1, latent_dim=100,
                 elayers=3, dlayers=3, clayers=3):
        ainputs = Input(shape=(timesteps, input_dim))
        cinputs = Input(shape=(timesteps, input_dim))

        # create encoding subnet
        encoded = LSTM(latent_dim, return_sequences=True)
        
#        for l in range(1, elayers):
#            encoded = LSTM(latent_dim)(encoded)
            
        # create decoding subnet
        decoded = RepeatVector(timesteps)(LSTM(latent_dim)(encoded(ainputs)))
        decoded = LSTM(input_dim, return_sequences=True)(decoded)

#        for l in range(1, dlayers):
#            decoded = LSTM(input_dim, return_sequences=True)(decoded)

        # set up autoencoder model
        self.autoencoder = Model(ainputs, decoded)

        # create and set up classifier
        classified = LSTM(128)(encoded(cinputs))

#        for l in range(1, clayers):
#            classified = LSTM(128)(classified)

        classified = Dense(3, activation='softmax')

        self.classifier = Model(cinputs, classified)

        # compile models
        self.autoencoder.compile(loss='binary_crossentropy',
                                 optimizer='adam')
        self.classifier.compile(loss='categorical_crossentropy',
                                optimizer='adam')

#    def __init__(self, auto, classif):
#        self.autoencoder = load_model(auto)
#        self.classifier = load_model(classif)
        
    def save(self, auto_out, classif_out):
        self.autoencoder.save_weights(auto_out)
        self.classifier.save_weights(classif_out)

    def fit_auto(X):
        self.autoencoder.fit(X, X)

    def fit_classif(X, y):
        self.classifier.fit(X, y)

    def evaluate(X, y):
        self.classifier.evaluate(X, y)
        
    def predict(X):
        return self.classifier.predict(X)

# a variational version
# TODO: make variational!
class VClassifier:
    def __init__(self, batch_size,
                 input_dim=1, intermediate_dim=50, latent_dim=100,
                 elayers=3, dlayers=3, clayers=3):
        ainputs = Input(shape=(timesteps, input_dim))
        cinputs = Input(shape=(timesteps, input_dim))

        # create encoding subnet
        encoded = LSTM(latent_dim)
        
        for l in range(1, elayers):
            encoded = LSTM(latent_dim)(encoded)
            
        # create decoding subnet
        decoded = RepeatVector(timesteps)(encoded(ainputs))
        decoded = LSTM(input_dim, return_sequences=True)(decoded)

        for l in range(1, dlayers):
            decoded = LSTM(input_dim, return_sequences=True)(decoded)

        # set up autoencoder model
        self.autoencoder = Model(ainputs, decoded)

        # create and set up classifier
        classified = LSTM(128)(encoded(cinputs))

        for l in range(1, clayers):
            classified = LSTM(128)(classified)

        classified = Dense(3, activation='softmax')

        self.classifier = Model(cinputs, classified)

        # compile models
        self.autoencoder.compile(loss='binary_crossentropy',
                                 optimizer='adam')
        self.classifier.compile(loss='categorical_crossentropy',
                                optimizer='adam')

    def __init__(self, auto, classif):
        self.autoencoder = load_model(auto)
        self.classifier = load_model(classif)
        
    def save(self, auto_out, classif_out):
        self.autoencoder.save_weights(auto_out)
        self.classifier.save_weights(classif_out)

    def fit_auto(X):
        self.autoencoder.fit(X, X)

    def fit_classif(X, y):
        self.classifier.fit(X, y)

    def evaluate(X, y):
        self.classifier.evaluate(X, y)
        
    def predict(X):
        return self.classifier.predict(X)
        

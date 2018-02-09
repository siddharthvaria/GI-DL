# KINDA DEPRECATED

from keras.models import Model, Sequential
from keras.layers import Input, LSTM

# vanilla autoencoder
class Autoencoder:
    def __init__(self):
        pass

# variational autoencoder
class VAE:
    def __init__(self):
        pass

# encoder subnet
# can be initialized solo or from the weights of a pretrained autoencoder
class Encoder:
    def __init__(self):
        pass

    def __init__(self, autoencoder, layers):
        for l in range(layers):
            pass

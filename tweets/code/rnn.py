import keras

from keras.models import Sequential

from keras.layers import Activation, Dense, Dropout
from keras.layers.recurrent import LSTM

from gensim import Word2Vec

def train_rnn(m, i):
	model = Word2Vec.load('tweets.model')

	nn = Sequential()

	nn.add(LSTM())
	nn.add(Dropout())

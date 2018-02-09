# main model
import pickle
import numpy as np

from numpy.random import random

import math

import spams
from spams import spams.omp

from keras.models import Sequential
from keras.layers import LSTM

from sklearn.decomposition import SparseCoder

print 'Starting up...'

# load preprocessed data
U = pickle.load(open('U.pkl', 'r')) # common embedding matrix
vocab = pickle.load(open('vocab.pkl', 'r')) # vocabulary
codes = pickle.load(open('word.codes', 'r')) # sparse codes for common words

w = []

#print U

#def sparse_loss(y_pred, w):
#    '''
#        Helper function. Custom loss function to be used in model.
#    '''
#    v = np.dot(U, y_pred) - w
#    v = v + alpha * np.sum(y_pred)
#    v = v + beta * abs(np.sum(y_pred) - 1)
#
#    return math.sqrt(np.dot(v, v))

print 'Initializing parameters...'

# initialize parameters
alpha = .002
beta = .2

max_len = 24

# compute codes
coder = SparseCoder(U, transform_algorithm='lasso_cd', transform_alpha = alpha, split_sign=True)

X = []
y = []
keys = []

for word in vocab:
#    if word.startswith('http://'):
#        continue
#
#    if word.startswith('https://'):
#        continue
    
    keys.append(word)
    X.append(random(2000))
    y.append(vocab[word])

print np.shape(U)
print np.shape(X)
    
X = coder.fit_transform(X, y)

print np.shape(X)

codes = {}

for k in range(len(keys)):
    codes[keys[k]] = X[k]
    
pickle.dump(codes, open('word.codes', 'w'))

#print codes['m.v.p.']
print keys[-1] + ': ' + str(codes[keys[-1]])
#print vocab[keys[-1]]

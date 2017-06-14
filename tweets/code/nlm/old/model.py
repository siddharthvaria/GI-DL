# main model
import pickle
import numpy as np

import math
from random import random

from keras.models import Sequential
from keras.layers import LSTM

from scipy.optimize import minimize

print 'Starting up...'

# load preprocessed data
U = pickle.load(open('U.pkl', 'r')) # common embedding matrix
vocab = pickle.load(open('../minivocab.pkl', 'r')) # vocabulary
codes = pickle.load(open('word.codes', 'r')) # sparse codes for common words

w = []

def sparse_loss(y_pred, w):
    '''
        Helper function. Custom loss function to be used in model.
    '''
    v = np.dot(U, y_pred) - w
    v = v + alpha * np.sum(y_pred)
    v = v + beta * abs(np.sum(y_pred) - 1)

    return math.sqrt(np.dot(v, v))

print 'Initializing parameters...'

# initialize parameters
alpha = 1
beta = .2

max_len = 24

# compute codes
starter = []

for i in range(8000):
    starter.append(random())
    
starter = np.array(starter)

print 'Computing sparse codes...'

i = 1

for word in vocab:
    # find optimal code
    w = vocab[word]

    if word in codes:
        continue
    
    def callbackf(params):
        print params
        print sparse_loss(params, w)
    
    code = minimize(sparse_loss, starter, args=(w), callback=callbackf, options={'maxiter':10})
    codes[word] = code

    if i % 100 == 0:
        print str(i) + ' codes computed'
        break # REMOVE AFTER TESTING

    i += 1

pickle.dump(codes, open('word.codes', 'w'))

print codes['m.v.p.']

# main model
import pickle
import numpy as np

from numpy.random import random

import math

from keras.models import Sequential
from keras.layers import LSTM

from sklearn.decomposition import SparseCoder

import sys

args = sys.argv
alg = 'lasso_cd' if len(args) < 2 else args[1]
alpha = .002 if len(args) < 3 else args[2]

def generate(alg='lasso_cd', alpha=.002, d=100, verbose=True):
    print 'Starting up...'

    # load preprocessed data
    U = pickle.load(open('U.pkl', 'r')) # common embedding matrix
#    vocab = pickle.load(open('vocab.pkl', 'r')) # vocabulary
    vocab = pickle.load(open('minivocab.pkl', 'r')) # vocabulary    
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

    if verbose:
        print 'Initializing parameters...'

    # initialize parameters
    beta = .2

    max_len = 24

    # compute codes
    coder = SparseCoder(U, transform_algorithm=alg,
                        transform_alpha = alpha)#, split_sign=True)

    X = []
    y = []
    keys = []

    if verbose:
        print 'Generating data arrays'
    
    for word in vocab:
    #    if word.startswith('http://'):
    #        continue
    #
    #    if word.startswith('https://'):
    #        continue
    
        keys.append(word)
        X.append(random(d))
        y.append(vocab[word])

    

    if verbose:
        print np.shape(U)
        print np.shape(X)
        
    X = coder.fit_transform(X, y)

    if verbose:
        print np.shape(X)
        
    codes = {}

    for k in range(len(keys)):
        codes[keys[k]] = X[k]
    
    pickle.dump(codes, open('word.codes', 'w'))

    if verbose:
        #print codes['m.v.p.']
        print keys[-1] + ': ' + str(codes[keys[-1]])
        #print vocab[keys[-1]]

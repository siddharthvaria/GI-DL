# preprocessing
import keras
#from keras import LSTM

import operator
import pickle
import numpy as np

from gensim.models import Word2Vec

def preprocess(v = 250, uout = 'U.pkl', wout = 'word.codes', cout = 'common.words', verbose=True):
    # fixed number of common words
    vocab = pickle.load(open('counts.pkl', 'r'))
    embeddings = pickle.load(open('vocab.pkl', 'r'))

    codes = {}

    # extract counts
    counts = sorted(vocab.items(), key=operator.itemgetter(1))
    common = counts[-v:]

    if verbose:
        print common[:100]

    mini = counts[-v - 20000:-v]

    minivocab = {}

    for m in mini:
        if m[0] not in embeddings.keys():
#             print m
            continue
    
        minivocab[m[0]] = embeddings[m[0]]

    if verbose:
        print len(minivocab.keys())
    
    pickle.dump(minivocab, open('minivocab.pkl', 'w'))

    model = Word2Vec.load('tweets.model')

    l = 100

    U = np.zeros((v, l))

    # generate U
    for i in range(v): # iterate over common words
        vec = model[common[i][0]] if common[i][0] in model else np.zeros(l)
    
        for j in range(l): # iterate over embedding indices
            U[i][j] = vec[j]

        codes[common[i][0]] = np.zeros((v, 1))
        codes[common[i][0]][j] = 1

    pickle.dump(U, open(uout, 'w'))
    pickle.dump(codes, open(wout, 'w'))

    cwords = {}

    for count in common:
        cwords[count[0]] = embeddings[count[0]]
    
    pickle.dump(cwords, open(cout, 'w'))

#    print np.shape(U)
#    print np.shape(codes)

def extract_data(dataf, timesteps = 140):
    lines = dataf.read_lines()

    X = []
    y = []

    for line in lines:
        splits = line.split('\t')

        x = splits[1]

        while len(x) < timesteps:
            x.append(0)

        X.append(x)
        y.append(1 if label == 'aggress' else (-1 if label == 'loss' else 0))

    return X, y

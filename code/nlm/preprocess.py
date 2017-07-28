# preprocessing
import keras
#from keras import LSTM

import operator
import pickle
import numpy as np
import random

from gensim.models import Word2Vec

def preprocess(v = 2000, uout = 'U.pkl', wout = 'word.codes', cout = 'common.words'):
    # fixed number of common words
    vocab = pickle.load(open('counts.pkl', 'r'))
    embeddings = pickle.load(open('vocab.pkl', 'r'))

    codes = {}

    # extract counts
    counts = sorted(vocab.items(), key=operator.itemgetter(1))
    common = counts[-v:]

    print common[:100]

    minicab = counts[-v - 5000:-v]

    minivocab = {}

    for m in minicab:
        if m[0] not in embeddings.keys():
#            print m
            continue
    
        minivocab[m[0]] = embeddings[m[0]]

    print len(minivocab.keys())
    
    pickle.dump(minivocab, open('minivocab.pkl', 'w'))

    model = Word2Vec.load('../tweets.model')

    l = 100

    U = np.zeros((l, v))

    # generate U
    for i in range(v):
        vec = model[common[i][0]] if common[i][0] in model else np.zeros(l)
    
        for j in range(l):
            U[j][i] = vec[j]

        codes[common[j][0]] = np.zeros((l, 1))
        codes[common[j][0]][j] = 1

    pickle.dump(U, open(uout, 'w'))
    pickle.dump(codes, open(wout, 'w'))
    pickle.dump(common, open(cout, 'w'))

def extract_data(dataf, timesteps = 140, use_y=True,
                 label_split=2, text_split=1,
                 cap=None, separator='\t', read=None):
    print '[extract] Starting...'

    lines = open(dataf, 'r').readlines()

    if read=='irregular':
        tabs = open(dataf, 'r').read().split('\t')

        lines = []
        line = []

        for i in range(8, len(tabs)):
            if i > 0 and i % 7 == 6:
                lines.append(unicode('\t'.join(line)))
#                print line
                line = []
            else:
                line.append(unicode(tabs[i].lower()))

    X = []
    y = []

    l = 0

    label = 0

    print '[extract] Lines: ' + str(len(lines))

    for line in lines:
        lwr = line.lower()

        splits = lwr.split(separator)

        if len(splits) == 0:
            print line
            continue

        if len(splits) == 1:
            print line
            continue

        x = splits[text_split]

        if use_y:
            label = splits[label_split]
#            print label
        
        cs = []

        for c in x:
            cs.append([ord(c), 0])

        while len(cs) < timesteps:
            cs.append([0, 0])

        if len(cs) > timesteps:
            cs = cs[:timesteps]

        X.append(cs)

#        print cs

#        print len(cs)
        if use_y:
            if 'aggress' in label or 'insult' in label or 'brag' in label or\
                'hypervigilance' in label or 'authority' in label:
                y.append(2)
            elif 'loss' in label or 'grief' in label or 'distress' in label or\
                'sadness' in label or 'loneliness' in label or\
                'death' in label:
                y.append(0)
            else:
                y.append(1)

#        print l
            
#        if cap and l > cap:
#                break

        l += 1
            
    if cap:
        uncapped = X
        X = []

        for i in range(cap):
            r = random.randint(0, len(uncapped)-1)
            X.append(uncapped[r])
            del uncapped[r]

    X = np.array(X)
    if use_y:
        y = np.array(y)
        return X, y
    
    return X

def data_split(dataf, label_split, text_split,
               val_size = .15, test_size = .15, outf=None):
    

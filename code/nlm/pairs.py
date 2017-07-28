import pickle
import math
import numpy as np

def generate():
#    vocab = pickle.load(open('vocab.pkl', 'r'))
    vocab = pickle.load(open('minivocab.pkl', 'r'))
    codes = pickle.load(open('word.codes', 'r'))
    U = pickle.load(open('U.pkl', 'r'))
    common = pickle.load(open('common.words', 'r'))
    
    letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
            'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    
    #print codes
    
    print 'Finding spelling pairs...'
    
    # find common pairs
    print 'Checking common words'
    '''
    pairs = {}
    
    for i in range(len(common)):
    c = common[i][0]
    pairs[c] = []
    
    #    print c
    
    for word in vocab:
        if not word[0].isalpha():
            continue

        if word.startswith('https://'):
            continue
        
        if word not in codes:
            continue

        if word == c:
            continue

        if codes[word][i] > 0.25:
            pairs[c].append(word)
            print c + ', ' + word

    if i % 1000 == 0:
        print 'Checked ' + str(i + 1) + ' common words'

    pickle.dump(pairs, open('common.pairs', 'w'))'''

    # split all words into buckets by first character
    buckets = {}
    
    for c in letters:
        buckets[c] = []

    for word in vocab:
        if word in codes and word[0].isalpha():# and not word.startswith('http'):
            buckets[word[0]].append(word)

    #    else:
    #        print word

    # find uncommon pairs
    count = 1

    pairs = {}

    #for b in letters:
    for b in ['a']:
        bucket = buckets[b]
        print 'Checking bucket ' + str(b)

    #    print bucket
    
    #    pairs = {}
    
        for word in bucket:
            pairs[word] = []
    
        for pair in bucket: # for each start word
            for word in bucket: # for each subsequent word
                if word == pair:
                    continue

                u = codes[pair]
                v = codes[word]
            
#            d = math.sqrt(np.sum(u * v))

#            d = np.dot(u, v)
#            d = d/(math.sqrt(np.dot(u, u)) * math.sqrt(np.dot(v, v)))

                d = math.sqrt(np.dot(u - v, u - v))

#            print d
            
#            if math.sqrt(np.sum(codes[pair] * codes[word])) > .5:
#                pairs[pair].append(word)

                if d < .2:
                    pairs[pair].append(word)

                if count % 1000000 == 0:
                    print str(count) + ' pairs checked'

                count += 1

#    pickle.dump(pairs, open(str(b) + '.pairs', 'w'))

    pickle.dump(pairs, open('word.pairs', 'w'))

    print pairs

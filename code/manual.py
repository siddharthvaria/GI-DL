import pickle
import pairs

def print_pair(pair, vocab):
    w1 = pair[0]
    w2 = pair[1]

    if not w1 in vocab or not w2 in vocab:
        print 'Pair not found: ' + w1 + ', ' + w2
        return
    
    cdist = pairs.dist(vocab[w1], vocab[w2])
    edist = pairs.edit_dist(w1, w2)

    print 'Pair: ' + w1 + ', ' + w2 + '; distances: ' + str(cdist) + ', ' + str(edist)

vocab = pickle.load(open('vocab.pkl', 'r'))

p = [('aint', 'ain\'t'), ('blue', 'red'), ('sky', 'fire'), ('fire', 'lit'), ('gun', 'run'), ('opp', 'pop'), ('block', 'street'), ('dog', 'dawg')]

misspellings = [('aggressive', 'agressive'), ('aggression', 'agression'), ('believe', 'belive'), ('believe', 'beleive'), ('definitely', 'definately'), ('disappear', 'dissapear'), ('finally', 'finaly'), ('piece', 'peice'), ('really', 'realy'), ('tattoo', 'tatoo'), ('truly', 'truely'), ('until', 'untill')]

print 'Miscellaneous pairs.'

for pair in p:
    print_pair(pair, vocab)

print '\nMisspelled pairs.'
    
for pair in misspellings:
    print_pair(pair, vocab)

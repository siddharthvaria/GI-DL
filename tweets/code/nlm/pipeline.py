import pickle

import embedder
import preprocess
import skmodel
import pairs

#import classify_driver

#vfile = 'vocab.pkl'
vfile = 'minivocab.pkl'
vout = 'paired_vocab.pkl'

def pipeline(d=100, win=7, mcount=2, wkrs=4,
             v=2000,
             alg='lasso_cd', alpha='.002'):
    # run w2v model to generate embedding model
    d = 100
    win = 7
    mcount = 2
    wkrs = 4

    #embed(d, win, mcount, wkrs)

    # preprocess vocab
    v = 2000
    
    preprocess.preprocess(v)
    
    # run skmodel to generate sparse codes
    skmodel.generate(alg, alpha)
    
    # run pairs to find pairs
    p = pairs.generate()

    # substitute words in vocab
    vocab = pickle.load(open(vfile, 'r'))
    vocab = pairs.substitute(p, vocab)
    pickle.dump(vocab, open(vout, 'w'))
    
    # plug pairs into classifier... somehow
    classifier = classifier.Classifier()

vs = [1000, 2000, 5000]
algs = ['lasso_lars', 'lasso_cd', 'lars', 'threshold']

for v in vs:
    for alg in algs:
        print 'Running with common vocabulary size ' + str(v) + ' and algorithm ' + alg
        pipeline(v=v, alg=alg)

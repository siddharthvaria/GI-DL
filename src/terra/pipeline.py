import pickle

import w2v
import preprocess
import skmodel
import pairs

import classify_driver

#vfile = 'vocab.pkl'
vfile = 'minivocab.pkl'
vout = 'paired_vocab.pkl'

def pipeline(d=100, win=7, mcount=2, wkrs=4,
             v=2000,
             alg='lasso_cd', verbose=False, alpha=.001,
             threshold=.2):
    print 'Running with parameters v=' + str(v) + ', alg=' + alg + ', threshold=' + str(threshold)
    
    # run w2v model to generte embedding model
#    d = 100
#    win = 7
#    mcount = 2
#    wkrs = 4

#    w2v.embed(d, win, mcount, wkrs)

    # preprocess vocab
    print 'Preprocessing...'
    
    v = 2000
    
    preprocess.preprocess(v, verbose=verbose)
    
    # run skmodel to generate sparse codes
    print 'Generating codes...'
    skmodel.generate(alg, d=d, verbose=verbose)
    
    # run pairs to find pairs
    print 'Finding pairs...'
    
    name = 'w2v:' + str(d) + '-' + str(win) + '-' + str(mcount) + '-' + str(wkrs)
    name = name + '_preprocess:' + str(v)
    name = name + '_codes:' + alg + '-' + str(alpha)
    
    p = pairs.generate(name=name, thresh=threshold, verbose=verbose)

    # substitute words in vocab
    print 'Substituting vocab...'
    vocab = pickle.load(open(vfile, 'r'))
    print len(vocab)
    vocab = pairs.substitute(p, vocab)
    pickle.dump(vocab, open(vout, 'w'))
    
    # plug pairs into classifier... somehow
    #classifier = classifier.Classifier()
    print 'Classifying...'
    return classify_driver.drive()

ds  = [100, 200, 300]
wins = [3, 5, 7, 9, 11]
mins = [1, 2, 3]
vs = [50, 100, 200, 500]
algs = ['threshold']#, 'lasso_cd', 'lasso_lars', 'lars']#, 'threshold']
#tols = [.0001, .001, .01, .1]
ts = [.01, .1, .2, .5]

results = {}

#for d in ds:
#    for w in wins:
#        for m in mins:
#            for v in vs:
#                for alg in algs:
#                    print 'Running with common vocabulary size ' + str(v) + ' and algorithm ' + alg
#                    results[(d, w, m, v, alg)] = pipeline(v=v, alg=alg)

for v in vs:
    for alg in algs:
        for t in ts:
            print 'Running with common vocabulary size ' + str(v) + ' and algorithm ' + alg + ' and threshold ' + str(t)
            
            results[(v, alg, t)] = pipeline(v=v, alg=alg,
                                            threshold=t)

print results
open('parameter.results', 'w').write(str(results))

import pickle
import numpy as np

def test_contains(f1, f2, i1, i2, s1='\t', s2='\t'):
    tweets = []

    for line in open(f2).readlines()[1:]:
        if line == '' or len(line.split(s2)) == 1:
            continue
        
#        print line
        tweets.append(line.split(s2)[i2].strip().lower())

    large = []

    lines = []
    tabs = open(f1, 'r').read().split(s1)
    line = []

    for i in range(8, len(tabs)):
        if i > 0 and i % 7 == 6:
            lines.append('\t'.join(line))
            line = []
        else:
            line.append(tabs[i].lower())
    
    for line in lines[1:]:
        if len(line.split(s1)) < 2:
            continue
        
        large.append(line.split(s1)[i1].strip().lower())

    contains = []

    for t in tweets:
        if t in large:
            contains.append(1)
        else:
            contains.append(0)

    print tweets
    print large

    print len(large)

    print np.sum(contains)
    return contains

test_contains('../../tsv/all_labeled.tsv', '../../tsv/train.tsv', 2, 1)

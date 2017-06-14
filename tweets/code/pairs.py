'''
Finds pairs of words that could be misspellings of each other
'''
import numpy as np
import pickle

import sys

args = sys.argv()

def pair_dists(w1, w2, vfile='vocab.pkl'):
        vocab = pickle.load(open(vfile, 'r'))

        cdist = dist(vocab[w1], vocab[w2])
        edist = edit_dist(w1, w2)

        return cdist, edist

def find_pairs(vfile='vocab.pkl'):
	'''
	    Finds pairs of words similar to each other in both distance and
            edit distance.
	'''
        print 'Starting find_pairs()...'
        print 'Loading vocabulary...'
        
	vocab = pickle.load(open(vfile, 'r'))

	pairs = []

	# TODO: find proper values
	t = .3
	e = 3

        print 'Loaded. Size of vocabulary: ' + str(len(vocab))

        print 'Bucketting...'
        
        buckets = split_buckets(vocab)

        print 'Generating pairs...'
        
        for b in range(26):
                bucket = buckets[b]

                print 'On \'' + chr(b + 97) + '\''
                
	        for word in bucket:
		        for other in bucket:
			        if word == other:
				        continue
                                if word[0] != other[0]:
                                        continue
                                if abs(len(word) - len(other)) > e:
                                        continue
			        if (other, word) in pairs:
				        continue
			        if dist(vocab[word], vocab[other]) < t:
				        pairs.append((word, other))
                                        
        print str(len(pairs)) + ' pairs generated.'
                                        
	spairs = []

        print 'Checking edit distance...'

	for pair in pairs:
		if edit_dist(pair[0], pair[1]) < e:
			spairs.append(pair)

        print 'Done.'

	return spairs

def split_buckets(vocab):
        buckets = []

        for i in range(26):
                buckets.append([])

        buckets.append([])

        for word in vocab:
                o = ord(word[0])

                if o < 97 or o > 122:
                        buckets[26].append(word)
                        continue
                
                buckets[o - 97].append(word)

        return buckets

def dist(v1, v2, cosine=True):
	'''
	    Helper function. Computes distance between two word embeddings.
	'''
	if cosine:
		return abs(np.sum(np.multiply(v1, v2))/(np.linalg.norm(v1) * np.linalg.norm(v2)))
		
	return np.linalg.norm(v2 - v1)

def edit_dist(w1, w2, type='lcs'):
	'''
	    Helper function. Computes edit distance between two words.
	'''

	dist = []

	m = len(w1)
	n = len(w2)

        if type == 'levenshtein':
                # initialization

	        for i in range(m):
		        dist.append([])

		        for j in range(n):
			        dist[i].append(0)

		        dist[i][0] = i

	        for j in range(n):
		        dist[0][j] = j

	        # recurrence

	        for i in range(m):
		        for j in range(n):
			        dist[i][j] = min((dist[i - 1][j] + 1,
							 dist[i][j - 1] + 1,
							 dist[i - 1][j - 1] + 0 if w1[i] == w2[j] else 2))

	        return dist[m - 1][i - 1]

        elif type == 'simple':
                # initialization

	        for i in range(m):
		        dist.append([])

		        for j in range(n):
			        dist[i].append(0)

		        dist[i][0] = i

	        for j in range(n):
		        dist[0][j] = j

	        # recurrence

	        for i in range(m):
		        for j in range(n):
			        dist[i][j] = min((dist[i - 1][j] + 1,
							 dist[i][j - 1] + 1,
							 dist[i - 1][j - 1] + 0 if w1[i] == w2[j] else 2))

	        return dist[m - 1][i - 1]

        elif type == 'lcs':
                lcs = []

                for i in range(m):
                        lcs.append([])
                        
                        for j in range(n):
                                lcs[i].append(0)

                for i in range(1, m):
                        for j in range(1, n):
                                if w1[i] == w2[j]:
                                        lcs[i][j] = lcs[i - 1][j - 1] + 1
                                else:
                                        lcs[i][j] = max([lcs[i][j - 1],
                                                         lcs[i - 1][j]])

                return lcs[m - 1][n - 1]
	
def min(a):
	'''
	    Helper function. Returns minimum value in an array.
	'''
	min = a[0]

	for i in range(1, len(a)):
		if a[i] < min:
			min = a[i]

	return min

def max(a):
        max = a[0]

        for i in range(1, len(a)):
                if a[i] > max:
                        max = a[i]

        return max

#print 'SAMPLES'
#print 'word 1, word 2: distance, edit distance'
#print '(TODO)'

#find_pairs()

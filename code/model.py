from gensim.models import Word2Vec

import pickle

tweets = []

lines = open('../tsv/all.tsv', 'r').readlines()[1:]

counts = {}

out = open('../tsv/auto.tsv', 'w')

for line in lines:
#	print line
	l = line.split('\t')[6].lower()
	sentence = l.split()
	tweets.append(sentence)

        outline = ''

        for c in l:
                outline = outline + c + ' '

        outline = outline + '|'

        for c in l:
                outline = outline + ' ' + c
        
        out.write(outline + '\n')
        
model = Word2Vec(tweets, size=100, window=7, min_count=2, workers=4)

model.save('tweets.model')

vocab = {}

for line in lines:
	for w in line.split('\t')[6].split():
		word = w.lower()
		
		if word not in vocab and word in model:
			vocab[word] = model[word]

pickle.dump(vocab, open('vocab.pkl', 'w'))

for tweet in tweets:
        for word in tweet:
                if word.startswith('@'):
                        continue

                if word in vocab:
                        if word in counts:
                                counts[word] += 1
                        else: counts[word] = 1

pickle.dump(counts, open('counts.pkl', 'w'))

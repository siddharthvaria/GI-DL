'''
Seeded Clustering Algorithm for Gang Intervention research project
Author: Terra
'''

import nltk
import re
import csv
import math
import string
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestCentroid
from nltk.stem.porter import PorterStemmer



stemmer = PorterStemmer()

def stem_tokens(tokens, stemmer):
    stems = []
    for token in tokens:
        stems.append(stemmer.stem(token))
    return stems

def tokenize(data):
    tokens = nltk.word_tokenize(data)
    stems = stem_tokens(tokens, stemmer)
    return stems

def get_labeled(data_file, cutoff):
	f = open(data_file, 'rU')
	reader = csv.DictReader(f)
	tweets = []

	counts = {}

	for row in reader:
		c = (row['CONTENT'])
		label = row['LABEL'].lower().split(',')
		label = re.sub(r'\W+', '', label[0])
		tweets.append((c, label))
		try:
			counts[label]+=1
		except:
			counts[label]=1

	#verify each label occurs enough times
	invalid = []
	for c in counts:
		if counts[c] < cutoff:
			invalid.append(c)

	for t in tweets:
		if t[1] in invalid:
			tweets.remove(t)

	X, y = zip(*tweets)

	return X, y

def get_unlabeled(data_file):
	f = open(data_file, 'rU')
	reader = csv.DictReader(f)
	tweets = []

	for row in reader:
		tweet = row['CONTENT']
		tweets.append(tweet)
	
	return tweets

def preprocess(text):
	tokens = nltk.word_tokenize(text.decode('utf_8').lower())
	result = ''
	trigger = False
	for t in tokens:
		if t.startswith('@'):
			trigger = True
		if t in string.punctuation or trigger: 
			continue
		else:
		    result = result + ' ' +t
	return result.strip()


#TODO: Should we be fitting to combined dataset or just labeled???
def tf_idf(labeled, unlabeled):
	labeled_data = []
	unlabeled_data = []

	for l in labeled:
		#note that this will strip + out of emoji encoding
		t = preprocess(l)
		labeled_data.append(l)

	for u in unlabeled: 
		t = preprocess(u)
		unlabeled_data.append(t)
	
	combined_data = labeled_data+unlabeled_data

	#TODO get better stop_word list for dataset with Maha
	tfidf = TfidfVectorizer(tokenizer=tokenize, stop_words='english', decode_error='ignore')
	tfidf.fit(labeled_data)
	a = tfidf.transform(labeled_data)
	b = tfidf.transform(unlabeled_data)

	return a.todense(), b.todense()


#Variables!!
labeled_data_file = 'data/gakirah_labeled.csv'
unlabeled_data_file = 'data/top_communicators_unlabeled.csv'
output_file = 'data/top_communicators_labeled.csv'
label_size_cutoff = 10
#threshold = ???



#Get data from file
labeled_data, y = get_labeled(labeled_data_file, label_size_cutoff)
unlabeled_data = get_unlabeled(unlabeled_data_file)
print 'retrieved data'

#Get tf-idf of data
train_X, test_X = tf_idf(labeled_data, unlabeled_data)
print 'calculated tf-idf'


#get centroids of labeled clusters and variance
clf = NearestCentroid(metric='euclidean')
clf.fit(train_X, y)
print 'found labeled clusters'

#add each unlabeled data point to the best cluster 
#(without updating centroid) + variance
labels_for_communicators = clf.predict(test_X)
result_data = []
for i in range(0, len(labels_for_communicators)):
	result_data.append({'CONTENT': unlabeled_data[i], 'LABEL': labels_for_communicators[i]})
print 'assigned labels to unlabeled tweets'

#now add each unlabeled data point to best cluster 
#(with updating centroid) + variance

#save two sets of labelled data to file
w = open(output_file, 'w')
writer = csv.DictWriter(w, ['CONTENT', 'LABEL'])
writer.writeheader()
for c in result_data:
	writer.writerow(c)


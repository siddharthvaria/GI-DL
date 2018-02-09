import csv
import sys
sys.path.append('code/emotion/')
sys.path.append('code/tools')

from emotion_scoring import EmotionScorer
#from tools import tweet_preprocessing2

datafile = 'data/emotion/error_analysis.csv'
f = open(datafile, 'rU')
reader = csv.DictReader(f)

scored = set([])
not_scored = set([])
correctly = set([])
incorrectly = set([])

#scored = []
#not_scored = []
#correctly = []
#incorrectly = []

for row in reader:
	data = row['Wi: Words are scored?'].split(';')
	for d in data:
		try:
			w = d.split()[0]
			scored.add(w)
#			scored.append(w)
		except:
			continue

	data = row['Wi: Should be scored but aren\'t?'].split(';')
	for d in data:
		try:
			w = d.split()[0]
			not_scored.add(w)
#			not_scored.append(w)
		except:
			continue

	data = row['Wi: correct scoring?'].split(';')
	for d in data:
		try:
			w = d.split()[0]
			correctly.add(w)
#			correctly.append(w)
		except:
			continue

	data = row['Wi: incorrect scoring?'].split(';')
	for d in data:
		try:
			w = d.split()[0]
			incorrectly.add(w)
#			incorrectly.append(w)
		except:
			continue

print len(scored), len(not_scored), len(correctly), len(incorrectly)	
print 'Words scored: '+str(len(scored)/float(len(scored)+len(not_scored)))
print 'Words scored correctly: '+str(len(correctly)/float(len(correctly)+len(incorrectly)))
print 'Words scored incorrectly: '+str(len(incorrectly)/float(len(correctly)+len(incorrectly)))
f.close()

'''
datafile = 'data/gakirah/gakirah_aggress_loss.csv'
f = open(datafile, 'rU')
reader = csv.DictReader(f)

scorer = EmotionScorer()
stopwords = scorer.import_stopwords('data/emotion/aave_stopwords.txt')
scored = 0
not_scored = 0

for row in reader:
	words = tweet_preprocessing2(row['CONTENT'])
	for word in words:
		s = scorer.abs_scoring(word)
		if s and word not in stopwords: scored += 1
		else: not_scored += 1

print 'Words scored after improvements: '+str(scored/float(scored+not_scored))
'''
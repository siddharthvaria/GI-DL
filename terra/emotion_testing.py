from emotion_scoring import EmotionScorer
import csv
import math
import sys
sys.path.append('code/tools')
from tools import tweet_preprocessing

scorer = EmotionScorer(sw=True)
total_words_scored = 0
total_words = 0

data_file = 'data/gakirah/gakirah_aggress_loss.csv'
emoji_data_file = 'data/emotion/emoji_lex.txt'
f = open(data_file, 'rU')
reader = csv.DictReader(f)


#FOR EMOTION ERROR ANAYSIS
count = 1
for row in reader:
	if count > 50: break
	tweet = row['CONTENT']
	print str(count)+' '+tweet
	tokens = tweet_preprocessing(tweet)
	count += 1
	for t in tokens:
		res = scorer.min_max_score(t.lower())
		print t+' '+str(res)
	cont = raw_input()
	print


'''
#FOR EMOJI TESTING
emojis = {}
f = open(emoji_data_file, 'r')

for line in f:
	if len(line.strip()) < 1: continue
	tokens = line.split(' : ')
	emojis[tokens[0].lower()] = tokens[1]

f.close()

for e in emojis.keys():
	score = scorer.abs_scoring(e)

	print e+' : '+str(score)+' ('+str(scorer.in_DAL(emojis[e].strip()))+' '+emojis[e].strip()+')'
	print
'''



'''
for row in reader:
	tweet = row['CONTENT']
	#NEED TO FIX TO WORK WITH CHANGES
	ee, aa, ii, norm, words_scored, num_words = scorer.score(tweet, True)
	total_words += num_words
	total_words_scored += words_scored
	print str(ee)+' '+str(aa)+' '+str(ii)+' '+str(norm)
	print str(words_scored)+' / '+str(num_words)
	print ''
print 'avg. hit rate = '+str(float(total_words_scored)/total_words)
'''

'''
in_file = 'data/emotion/emoji_dict.txt'
out_file = 'data/emotion/emoji_lexicon.txt'
f = open(in_file, 'r')

storage = []
for line in f:
	tokens = line.split(' : ')
	s = scorer.score(tokens[1].strip('\n')) 
	tokens.append(s)
	storage.append(tokens)

f.close()
f = open(out_file, 'w')
for s in storage:
	print s
	f.write(s[0]+' : '+s[1]+' : '+str(s[2][0])+' '+str(s[2][1])+' '+str(s[2][2])+' '+str(s[2][3])+'\n')
f.close()
'''
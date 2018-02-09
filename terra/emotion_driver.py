from emotion_scoring import EmotionScorer
import csv
import math
import sys
sys.path.append('code/tools')
from tools import tweet_preprocessing
from unicode_emojis import UNICODE_EMOJI 

scorer = EmotionScorer(sw=True)
total_words_scored = 0
total_words = 0

data_file = 'data/gakirah/gakirah_aggress_loss.csv'
emoji_data_file = 'data/emotion/emoji_lex.txt'
f = open(data_file, 'rU')
reader = csv.DictReader(f)

'''
#FOR EMOTION ERROR ANAYSIS
count = 1
for row in reader:
	if count > 50: break
	tweet = row['CONTENT']
	print str(count)+' '+tweet
	tokens = tweet_preprocessing(tweet)
	count += 1
	for t in tokens:
		res = scorer.avg_score(t.lower(), _wiktionary=True)
		print t+' '+str(res)
	cont = raw_input()
	print
'''

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

reload(sys)
sys.setdefaultencoding("utf-8")

#SCORING Statistics
dev_file = 'data/_dev/dev_full.csv'
reader = csv.DictReader(open(dev_file, 'rb'))

non_std = {}
emoji_codes = UNICODE_EMOJI.keys()
found_count = 0
sw = scorer.import_stopwords('data/emotion/aave_stopwords.txt')
for row in reader:
	tweet = row['CONTENT']
	
	tokens = tweet_preprocessing(tweet)
	for t in tokens:
		if not scorer.in_DAL(t) and not scorer.in_wordnet(t) and t not in emoji_codes and t not in sw: 
			try: 
				non_std[t] += 1
			except:
				non_std[t] = 1
print 'found nonstandard'


n_scores = {}
print len(non_std.keys())
for n in non_std.keys():
	definition = scorer.translate(n, wiktionary=False, lexicons=True)
	if len(definition) < 1 or (len(definition) == 1 and len(definition[0].strip()) == 0): continue
	found_count += non_std[n]
	print n.encode('utf-8')+': '+str(definition)
	cont = raw_input()
	if cont == 'y' or cont == 'Y': n_scores[n] = (True, non_std[n])
	else: n_scores[n] = (False, non_std[n])

scored = 0
for n in n_scores:
	if n_scores[n][0]: scored += n_scores[n][1]

print '% non_std tokens scored by lex: '+str(float(found_count)/sum([non_std[n] for n in non_std.keys()]))
print '% of the tokens scored that are scored correctly: '+str(float(scored)/sum([n_scores[n][1] for n in n_scores.keys()]))
	




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
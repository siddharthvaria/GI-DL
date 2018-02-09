import csv
import operator
import re
import sys
sys.path.append('code/tools')
sys.path.append('code/emotion')
from tools import tweet_preprocessing_sw
from unicode_emojis import UNICODE_EMOJI 


#count tokens
files = ['data/_train/train_full.csv', 'data/_dev/dev_full.csv', 'data/_test/test_full.csv']
emojis = {}

for f in files:
	reader = csv.DictReader(open(f, 'rb'))
	for row in reader:
		text = row['CONTENT'].lower()
		tokens = text.strip().split()
		for t in tokens:
				try: 
					emojis[t] = emojis[t] + 1
				except:
					emojis[t] = 1

sorted_emojis = sorted(emojis.items(), key=operator.itemgetter(1))

for e in sorted_emojis:
	print e[0]+', '+str(e[1])
print '(sorted from least to most common)'

reload(sys)
sys.setdefaultencoding("utf-8")

#count percentage of emojis in nonstopwords
sw_file = 'data/emotion/aave_stopwords.txt'

sw = []
for l in open(sw_file, 'rb').readlines():
	sw.append(l.strip())
sw.append('::emoji::')
sw.append('USER_HANDLE')
sw.append('URL')

emoji_codes = UNICODE_EMOJI.keys()

count = 0
total_count = 0
for f in files:
	reader = csv.DictReader(open(f, 'rb'))
	for row in reader:
		text = row['CONTENT'].lower()
		tokens = tweet_preprocessing_sw(text, sw)
		for t in tokens:
			if t in emoji_codes: count +=1
			else: print t 
			total_count += 1

print '% emojis = '+str(float(count)/total_count)

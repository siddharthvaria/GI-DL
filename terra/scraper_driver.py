from tools import scraper, space_emoji
import sys
import csv
import re
from nltk.tokenize import TweetTokenizer
import sets


#reload(sys)
#sys.setdefaultencoding("utf-8")

test_url = 'http://twitter.com/Mr10kpooh/statuses/444706823451709440'
emoji_file = 'data/unicode_to_emoji.txt'
input_files = ['aintyoubecky_gkirahassassin_march_15_to_april_17_2014',
				'arrogant_bubba_march_15_to_april_17_2014',
				'bricksquadmain_march_15_to_april_17_2014', 
				'cashout063_march_15_to_april_17_2014',
				'dutchiedntshoot_march_15_to_april_17_2014',
				'fly_girllashea_march_15_to_april_17_2014',
				'lydiathebest__march_15_to_april_17_2014',
				'twdro_march_15_to_april_17_2014',
				'whynotchief_march_15_to_april_17_2014',
				'younggodumb_march_15_to_april_17_2014']
#input_files = [sys.argv[1],]

m = open(emoji_file, 'rU')
emoji_dict = {}
for line in m:
	tokens = line.split(' : ')
	emoji_dict[tokens[0]] = tokens[1].strip()

count1 = 0
count2 = 0

label_list = ['AUTHOR', 'CONTENT', 'ARTICLE_URL', 'PUBLISH_DATE', 'LABEL', 'DESCRIPTION', 'COLLAPSED CODES', 'EMOTIONAL SCORING']

for name in input_files:
	f = open('data/radian6/'+name+'.csv', 'rU')
	reader = csv.DictReader(f)

	w = open('data/radian6_results/'+name+'.csv', 'wb')
	writer = csv.DictWriter(w, label_list)
	writer.writeheader()

	for row in reader:
		if count1 + count2 % 1000 == 0:
			print count1+count2
			
#		if row['CONTENT'].lower() in all_tweets:
#			continue
		url = row['ARTICLE_URL']
		try: 
			tweet = scraper(url).decode('utf-8')
			tweet = space_emoji(tweet)

			print "retrieved: "+tweet
#			print old_tweet
			print

			row['CONTENT'] = tweet.encode('utf-8')
			row = {k: v for k, v in row.items() if k in label_list}

			writer.writerow(row)
#			all_tweets.add(tweet.encode('utf-8').lower())
#			all_tweets.add(old_tweet)
			count1 += 1
		except:
			tweet = row['CONTENT']
			print "couldn't get: "+tweet
			row = {k: v for k, v in row.items() if k in label_list}
			writer.writerow(row)
			count2 += 1

print 'count\'t get '+str(float(count2)/(count1+count2))+' of the tweets'

import re
import csv
import operator
import math


file_name = 'data/gakirah/gakirah_labeled.csv'
f = open(file_name, 'rU')
reader = csv.DictReader(f)

#LABEL SETS
aggress = ['aggress', 'threat', 'authority', 'insult', 'snitch', 'aware', 'guns']
loss = ['loss', 'grief', 'death', 'sad', 'alone']

try:
# UCS-4
    patt = re.compile(u'([\U00002600-\U000027BF])|([\U0001f300-\U0001f64F])|([\U0001f680-\U0001f6FF])')
except re.error:
# UCS-2
    patt = re.compile(u'([\u2600-\u27BF])|([\uD83C][\uDF00-\uDFFF])|([\uD83D][\uDC00-\uDE4F])|([\uD83D][\uDE80-\uDEFF])')

#key = emoji, value = counts
aggress_counts = {}
loss_counts = {}
other_counts = {}
total_counts = {}

#get counts 
for row in reader:
	found = False
	tweet = row['CONTENT']
	it = patt.finditer(tweet.decode('utf-8'))
	for res in it:
		try:
			total_counts[res.group()] += 1
		except:
			total_counts[res.group()] = 1

	for l in aggress:
		if l in row['LABEL'].lower():
			found = True
			it = patt.finditer(tweet.decode('utf-8'))
			for res in it:
				try:
					aggress_counts[res.group()] += 1
				except:
					aggress_counts[res.group()] = 1

	for l in loss:
		if l in row['LABEL'].lower():
			found = True
			it = patt.finditer(tweet.decode('utf-8'))
			for res in it:
				try:
					loss_counts[res.group()] += 1
				except:
					loss_counts[res.group()] = 1

	if not found:
		it = patt.finditer(tweet.decode('utf-8'))
		for res in it:
			try:
				other_counts[res.group()] += 1
			except:
				other_counts[res.group()] = 1

#get TFIDF scores
aggress_tfidf = {}
loss_tfidf = {}

for emoji in aggress_counts.keys():
	doc_count = 1
	if emoji in loss_counts.keys(): doc_count += 1
	if emoji in other_counts.keys(): doc_count += 1
	aggress_tfidf[emoji] = aggress_counts[emoji] * (math.log(3/float(doc_count)))

for emoji in loss_counts.keys():
	doc_count = 1
	if emoji in aggress_counts.keys(): doc_count += 1
	if emoji in other_counts.keys(): doc_count += 1
	loss_tfidf[emoji] = loss_counts[emoji] * (math.log(3/float(doc_count)))


#RESULTS
sorted_aggress = sorted(aggress_tfidf.items(), key=operator.itemgetter(1), reverse=True)
print 'top 10 agress by tf-idf:'
for item in sorted_aggress[:10]:
    print item[0]+' ('+str(item[1])+')'

sorted_loss = sorted(loss_tfidf.items(), key=operator.itemgetter(1), reverse=True)
print 'top 10 loss by tf-idf:'
for item in sorted_loss[:10]:
    print item[0]+' ('+str(item[1])+')'

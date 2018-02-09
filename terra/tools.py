import re
import csv
import sys
import urllib2 as ul
import HTMLParser
from string import punctuation

def unicode_csv_reader2(utf8_data, **kwargs):
    csv_reader = csv.DictReader(utf8_data, **kwargs)
    for row in csv_reader:
        yield {k:unicode(v, 'utf-8') for k, v in row.iteritems()}

def print_stats(sought_p, sought_r, sought_f1, nsought_p, nsought_r, nsought_f1, message):
	print message
	print 'sought precision: '+str(sought_p)
	print 'sought recall: '+str(sought_r)
	print 'sought f1 score: '+str(sought_f1)
	print
	print 'nsought precision: '+str(nsought_p)
	print 'nsought recall: '+str(nsought_r)
	print 'nsought f1 score: '+str(nsought_f1)
	print

def fscore(precision, recall):
    return 2*((precision*recall)/float(precision+recall))

def read_in_data(file):
	reader = unicode_csv_reader2(open(file, 'rb'))
	tweets = []
	for row in reader:
		if 'LABEL' in row.keys(): label = row['LABEL'].lower()
		else: label = ''

		tweet = row['CONTENT']
		tweets.append((tweet, label))

	print len(tweets)

	return tweets

def write_out_data(data, file):
	writer = csv.DictWriter(open(file, 'wb'), ['CONTENT', 'LABEL'])
	writer.writeheader()
	for d in data:
		writer.writerow({'CONTENT':d[0], 'LABEL': d[1]})
	return

def tweet_preprocessing(text, stopwords=[]):
	text = re.sub('(::emoji::)|#|', '', text.lower())
	text = re.sub('@[0-9a-zA-Z_]+', 'USER_HANDLE', text)
	text = re.sub('http://[a-zA-Z0-9_\./]*', 'URL', text)
	words = text.split(' ')
	if len(stopwords) > 0:
		words = [word.strip() for word in words if word not in stopwords]
		
	return words

def tweet_preprocessing_sw(text, stopwords):
	text = re.sub('(::emoji::)', '', text.lower())
	text = re.sub('@[0-9a-zA-Z_]+', 'USER_HANDLE', text)
	text = re.sub('http://[a-zA-Z0-9_\./]*', 'URL', text)
	words = text.split(' ')
	words = [word.strip() for word in words if word not in stopwords]

	return words

def scraper(url):
	h = HTMLParser.HTMLParser()
	webpage = ul.urlopen(url).read()
	regex = re.compile('<meta  property="og:description" content=".*">\n')
	r = regex.search(webpage)
	return h.unescape(r.group(0)[45:-6])

def space_emoji(data):
    if not data:
        return data
    if not isinstance(data, basestring):
        return data
    try:
    # UCS-4
        patt = re.compile(u'([\U00002600-\U000027BF])|([\U0001f300-\U0001f64F])|([\U0001f680-\U0001f6FF])')
    except re.error:
    # UCS-2
        patt = re.compile(u'([\u2600-\u27BF])|([\uD83C][\uDF00-\uDFFF])|([\uD83D][\uDC00-\uDE4F])|([\uD83D][\uDE80-\uDEFF])')
    return patt.sub(lambda m: ' '+m.group()+' ', data)

def get_emoji_dict():
	emoji_file = 'data/unicode_to_emoji.txt'
	m = open(emoji_file, 'rU')
	emoji_dict = {}
	for line in m:
		tokens = line.split(' : ')
		emoji_dict[tokens[0]] = tokens[1].strip()
	return emoji_dict

def get_label_sets():
	#LABEL SETS
#	aggress = ['aggress', 'insult', 'snitch', 'threat', 'brag', 'aod', 'aware', 'authority', 'trust', 'fight', 'identity', 'pride', 'power', 'lyric']      
	aggress = ['aggress', 'insult', 'snitch', 'threat', 'brag', 'aod', 'aware', 'authority', 'trust', 'fight', 'pride', 'power', 'lyric']
	loss = ['loss', 'grief', 'death', 'sad', 'alone', 'reac', 'guns']
	return aggress, loss

def get_other_labels():
	other = ['deleted/ sex', 'money', 'gen, rel', 'rel, wom', 'authenticity', 'anger', 'retweet', 'wom', 'convo/neighborhood', 'gen, money', 'gen/women', 'deleted', 'gen/location', 'rel', 'indentity', 'amiable?', 'happy', 'sex', 'promo', 'mention', 'gen, happy', 'general', 'gen', 'identity', 'rel/gen', 'convo', 'joke', 'trans', 'wom, rel']
	return other

def error_analysis(data_file, predictions, label, name):
	aggress, loss = get_label_sets()
	if label == 'aggress':
		pos_labels = aggress
	elif label == 'loss':
		pos_labels = loss
	else:
		pos_labels = loss+aggress

	data = read_in_data(data_file)
	if len(data) != len(predictions): 
		print "data does not match predictions"
		return

	both = zip(data, predictions)
	writer = csv.DictWriter(open(name+'.csv', 'wb'), ['CONTENT', 'LABEL', 'BIN_LABEL', 'PREDICTION'])
	writer.writeheader()

	for b in both:
		b_label = 0
		if b[0][1] in pos_labels: b_label = 1
		writer.writerow({'CONTENT': b[0][0], 'LABEL': b[0][1], 'BIN_LABEL': b_label, 'PREDICTION': b[1]})
	return




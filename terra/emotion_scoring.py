'''
EmotionScorer for Gang Intervention project
Author: Terra
Uses the DAL and various lexicons to get emotion scores (ee/pleasantness, aa/activation, and ii/imagery) for input text,
which are expected to be tweets (from gang members in Chicago).
'''

import sys
sys.path.append('code/tools/')

import math
from tools import tweet_preprocessing_sw
from nltk.corpus import wordnet as wn
from wordnik import *
import urllib2

apiUrl = 'http://api.wordnik.com/v4'
apiKey = '34f1b006d1202ececad7f0b92730a453b16a2ec6716c6a5a6'

class EmotionScorer:

	DAL_filepath = 'data/emotion/dict_of_affect.txt'
	lex_filepath = 'data/emotion/lexicon.txt'
	stopwords_filepath = 'data/emotion/aave_stopwords.txt'

	#returns the min, max of each of the 3 dimensions for tweet
	def min_max_score(self, text, _wordnet = True, _wiktionary = True, _lexicon = True, _sw = True):
		#variables
		ee_min, ee_max, aa_min, aa_max, ii_min, ii_max = 4, 0, 4, 0, 4, 0
		count = 0

		#preprocessing
		if _sw:
			t = tweet_preprocessing_sw(text, self.stopwords)
		else:
			t = tweet_preprocessing(text)

		text = self.translate(t, _wiktionary, _lexicon)

		#score text by getting min/max of each dimension over all words
		for t in text:
			n = self.in_wordnet(t)
			if t in self.dal.keys():
				score = self.dal[t]
				count +=1
			elif t.strip('s') in self.dal.keys():
				score = self.dal[t.strip('s')]
				count += 1
			elif _wordnet and len(n) > 0:
				text.extend(n)
				score = (0, 0, 0)
			else: score = (0, 0, 0)

			if score[0] > ee_max and score[0]<4: ee_max = score[0]
			if score[0] < ee_min and score[0]>0: ee_min = score[0]

			if score[1] > aa_max and score[1]<4: aa_max = score[1]
			if score[1] < aa_min and score[1]>0: aa_min = score[1]

			if score[2] > ii_max and score[2]<4: ii_max = score[2]
			if score[2] < ii_min and score[2]>0: ii_min = score[2]

		if count > 0: return ee_min, ee_max, aa_min, aa_max, ii_min, ii_max
		else: return (0, 0, 0, 0, 0, 0)

	'''
	#for justification of predictions
	def min_max_justification(self, word, _wordnet = True, _wiktionary = True, _lexicon = True, _sw = True):
		#variables
		ee_min, ee_max, aa_min, aa_max, ii_min, ii_max = 4, 0, 4, 0, 4, 0
		count = 0

		#preprocessing
		if _sw:
			t = tweet_preprocessing_sw(text, self.stopwords)
		else:
			t = tweet_preprocessing(text)

		text = self.translate(t, _wiktionary, _lexicon)

		#score text by getting min/max of each dimension over all words
		for t in text:
			n = self.in_wordnet(t)
			if t in self.dal.keys():
				score = self.dal[t]
				count +=1
			elif t.strip('s') in self.dal.keys():
				score = self.dal[t.strip('s')]
				count += 1
			elif _wordnet and len(n) > 0:
				text.extend(n)
				score = (0, 0, 0)
			else: score = (0, 0, 0)

			if score[0] > ee_max and score[0]<4: ee_max = score[0]
			if score[0] < ee_min and score[0]>0: ee_min = score[0]

			if score[1] > aa_max and score[1]<4: aa_max = score[1]
			if score[1] < aa_min and score[1]>0: aa_min = score[1]

			if score[2] > ii_max and score[2]<4: ii_max = score[2]
			if score[2] < ii_min and score[2]>0: ii_min = score[2]

		if count > 0: return ee_min, ee_max, aa_min, aa_max, ii_min, ii_max
		else: return (0, 0, 0, 0, 0, 0)
	'''

	#best value of text based on distance from 2 (values range from 1 to 3)
	def dev_score(self, text, _wordnet = True, _wiktionary = True, _lexicon = True, _sw = True):
#		ee_mean, aa_mean, ii_mean = 1.85, 1.67, 1.52
		ee_mean, aa_mean, ii_mean = 2, 2, 2
		ee, aa, ii = ee_mean, aa_mean, ii_mean
		count = 0

		#preprocessing
		if _sw:
			t = tweet_preprocessing_sw(text, self.stopwords)
		else:
			t = tweet_preprocessing(text)

		text = self.translate(t, _wiktionary, _lexicon)

		#score each word and save dim if if max dev from 2
		for t in text:
			n = self.in_wordnet(t)
			if t in self.dal.keys():
				score = self.dal[t]
				count +=1
			elif t.strip('s') in self.dal.keys():
				score = self.dal[t.strip('s')]
				count += 1
			elif _wordnet and len(n) > 0:
				text.extend(n)
				score = [ee_mean, aa_mean, ii_mean]
			else:
				score = [ee_mean, aa_mean, ii_mean]

			if math.fabs(ee_mean-score[0]) > math.fabs(ee_mean-ee): ee = score[0] 
			if math.fabs(aa_mean-score[1]) > math.fabs(aa_mean-aa): aa = score[1] 
			if math.fabs(ii_mean-score[2]) > math.fabs(ii_mean-ii): ii = score[2] 

		#return dimensions, norm
		if count > 0:
			return ee, aa, ii, self.norm(ee, aa, ii)
		else:
			return (0, 0, 0, 0)

	#gets the avg ee, aa, and ii over all words in tweet
	def avg_score(self, text, _wordnet = True, _wiktionary = False, _lexicon = True, _sw = True):
		#variables
		ee_sum, aa_sum, ii_sum = 0, 0, 0
		count = 0

		#preprocessing
		if _sw: t = tweet_preprocessing_sw(text, self.stopwords)
		else: t = tweet_preprocessing(text)
		text = self.translate(t, _wiktionary, _lexicon)

		#score each word and add to sum of resp. dimension

		for t in text:
			n = self.in_wordnet(t)
			if t in self.dal.keys():
				score = self.dal[t]
				count +=1
			elif t.strip('s') in self.dal.keys():
				score = self.dal[t.strip('s')]
				count += 1
			elif _wordnet and len(n) > 0:
				text.extend(n)
				score = [0, 0, 0]
			else:
				score = [0, 0, 0]

			ee_sum += score[0]
			aa_sum += score[1]
			ii_sum += score[2]

		if count == 0: count += 1
		#get avg of each dimension, norm
		ee = float(ee_sum)/count
		aa = float(aa_sum)/count
		ii = float(ii_sum)/count

		return ee, aa, ii, self.norm(ee_sum, aa_sum, ii_sum)

	#"translates" tweet based on the lexicons, wordnet, wiktionary
	def translate(self, text, wiktionary, lexicons):
		new_text = []

		for t in text:
			if lexicons and t in self.lex.keys() and t not in self.dal.keys():
				new_text.append(self.lex[t.encode('utf-8')].strip())
			elif wiktionary and t not in self.lex.keys() and t not in self.dal.keys():
				w = self.in_wiktionary(t)
				if len(w) > 0: new_text.extend(w)
			else:
				new_text.append(t)

		return new_text

	def translate_word(self, word, wiktionary = False, lexicon = True):
		if lexicon and word in self.lex.keys() and word not in self.dal.keys():
			print 'lex match found' #TESTING
			return self.lex[word].strip()
		elif wiktionary and word not in self.lex.keys() and word not in self.dal.keys():
			w = self.in_wiktionary(word)
			if len(w) > 0: return w 
		else: return ['']

	#calculate "norm" for given scores
	def norm(self, ee, aa, ii):
		norm = 0
		if ii != 0: norm = math.sqrt(math.pow(ee,2)+math.pow(aa,2))/ii
		return norm


	def in_wiktionary(self, word):
		upper = word.upper()
		gloss = ''
		try:
			definitions = self.wordnik.getDefinitions(word,
				sourceDictionaries='wiktionary', limit=1)
			u_definitions = self.wordnik.getDefinitions(upper,
				sourceDictionaries='wiktionary', limit=1)
			if definitions:
				gloss = definitions[0].text.lower().strip('.')
			elif u_definitions:
				gloss = u_definitions[0].text.lower()
			else: gloss = ''

			gloss = tweet_preprocessing_sw(gloss, self.stopwords)

			return gloss
		except urllib2.HTTPError, err:
			return ''


	def in_wordnet(self, word):
		try:
			syn = wn.synsets(word)
			for l in syn[0].lemmas():
				if l.name() in self.dal.keys():
					return l.name().split()
			return ''
		except:
			return ''


	def in_DAL(self, words):
		result = []
		for word in words.split():
			if word in self.dal and word not in self.stopwords:
				result.append(word)
		return result

	# stores DAL in dict with key = word and 
	# value = [ee score, aa score, ii score]
	def import_dal(self, filepath):
		dal = {}
		f = open(filepath, 'r')

		for line in f:
			tokens = line.split()
			dal[tokens[0]] = [float(tokens[1]), float(tokens[2]), float(tokens[3])]

		f.close()
		return dal

	# stores emojis as dict with key = unicode enocoding and 
	# value = gloss/description of emoji given by unicode
	def import_lex(self, filepath):
		emojis = {}
		f = open(filepath, 'r')

		for line in f:
			if len(line.strip()) < 1: continue
			tokens = line.split(' : ')
			emojis[tokens[0].lower()] = tokens[1]

		f.close()
		return emojis

	# stores stopwords in list 
	def import_stopwords(self, filepath):
		stopwords = []
		f = open(filepath, 'r')

		for line in f:
			stopwords.append(line.strip())

		f.close()
		return stopwords

	def __init__(self, sw=True):
		self.dal = self.import_dal(EmotionScorer.DAL_filepath)
		self.lex = self.import_lex(EmotionScorer.lex_filepath)
		client = swagger.ApiClient(apiKey, apiUrl)
		self.wordnik = WordApi.WordApi(client)
		if sw:
			self.stopwords = self.import_stopwords(EmotionScorer.stopwords_filepath)
		else:
			self.stopwords = []

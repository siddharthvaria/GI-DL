'''
Author: Terra
A utility class to allow a ML model to get features for a set of tweets
'''

import sys
sys.path.append('code/emotion/')
sys.path.append('code/tools')
sys.path.append('rjk2147/src')
from emotion_scoring import EmotionScorer
from tools import *
import nltk
import tagger as t

import embedder

import numpy as np

def train_tagger():
    # open training data
    infile = open("rjk2147/results/pos_tagged_4_fold_cv.txt", "r")
    train_sents = infile.readlines()
    infile.close()
    train_sents = train_sents[100:]
    # open CMU training data
    infile = open("rjk2147/data/gold/cmu_all_gold.txt")
    cmu_train_sents = infile.readlines()
    infile.close()

    tagger = t.tagger(brown_cluster_path='rjk2147/tools/TweeboParser/pretrained_models/twitter_brown_clustering_full',
                      #word2vec_path='../tools/word2vec/word2vec_twitter_model.bin',
                      #word2vec_path= '../tools/word2vec/glove.6B/glove.6B.300d.txt',
                      #word2vec_path= '../tools/word2vec/GoogleNews-vectors-negative300.bin',
                      wiktionary_path='rjk2147/data/wiktionary'
                      )
    window = tagger.window

    half_cmu_train_sents = cmu_train_sents[:len(train_sents)/2]
    infile.close()
    #

    all_sents = list()
    all_sents.extend(train_sents)
    all_sents.extend(cmu_train_sents)
    domain_list = None

    # Standard implementation of domain adaptation
    domain_list = ['*tweet*']*len(train_sents)
    domain_list.extend(['*cmu*']*len(cmu_train_sents))

    tagger.train(all_sents, domain_list)

    return tagger

# get unigrams of input text
def get_unigrams(tokens):
	feats = {}
	for token in tokens:
		try:
			feats[token]
		except:
			feats[token] = 0
		feats[token] += 1
	return feats

#gets bigrams of input text
def get_bigrams(tokens):
    tokens = ['*']+tokens+['STOP']
    result = {}
    bg = list(nltk.bigrams(tokens))
    for pair in bg:
        try:
            result[pair] += 1
        except:
            result[pair] = 0
    return result

def get_embeddings(tokens):
	e = embedder.Embedder()#model='common.words')

	embeddings = []
	for token in tokens:
		embeddings.append(e.embed(token))
        
	return embeddings

def get_embedding(tweet):
	e = embedder.Embedder()#model='common.words')

	return e.embed(tweet)

def get_wfeats(tweets):
	wfeats = []
	tfeats = []
    
	for tweet in tweets:
#        print tweet
		wfeats.append([e for e in get_embeddings(tweet.split()) if e != None])
		temb = get_embedding(tweet)
		tfeats.append(temb if temb != None else np.ones(100))
        
	return tfeats, wfeats

#adds emotion score results to a feature dict
def get_emotion(emotion):
	feats = {}
	if len(emotion) == 4:
		feats['ee'] = emotion[0]
		feats['aa'] = emotion[1]
		feats['ii'] = emotion[2]
		feats['norm'] = emotion[3]

	if len(emotion) == 6:
		feats['ee_min'] = emotion[0]
		feats['ee_max'] = emotion[1]
		feats['aa_min'] = emotion[2]
		feats['aa_max'] = emotion[3]
		feats['ii_min'] = emotion[4]
		feats['ii_max'] = emotion[5]

	return feats

def get_desc(tokens):
	feats = {}
	for token in tokens:
		try:
			feats['desc_'+token]
		except:
			feats['desc_'+token] = 0
		feats['desc_'+token] += 1
	return feats


#version of get feats to work with auto_classify.py
def get_feats(tweets, _unigrams = True, _bigrams = False, _postags = False, _embeddings = False, _emotion = False,
	description = False, pos_tagger=None):
	e_score = EmotionScorer()
	X = []
	index = 0

	for tweet in tweets:
		feats = {}
		tokens = tweet_preprocessing(tweet)
		if _unigrams:
			u = get_unigrams(tokens)
			for key in u.keys():
				feats[key] = u[key]
		if _bigrams:
			b = get_bigrams(tokens)
			for key in b.keys():
				feats[key] = b[key]
		if _postags and (_postags == 'u' or _postags == 'b'):
			p = get_postags(tokens, pos_tagger, _postags)
			for key in p.keys():
				feats[key] = p[key]
		if _embeddings:
			m = get_embeddings(tokens)
			for key in m.keys():
				feats[key] = m[key]
			
		if _emotion == 'avg/all':
			score = e_score.avg_score(tweet, _wiktionary=False)
			e = get_emotion(score)
			for key in e:
				feats[key] = e[key]
		elif _emotion == 'avg/ee':
			score = e_score.avg_score(tweet, _wiktionary=False)
			e = get_emotion(score)
			feats['ee'] = e['ee']
		elif _emotion == 'avg/norm':
			score = e_score.avg_score(tweet, _wiktionary=False)
			e = get_emotion(score)
			feats['norm'] = e['norm']
		elif _emotion == 'dev/all':
			score = e_score.dev_score(tweet, _wiktionary=False)
			e = get_emotion(score)
			for key in e:
				feats[key] = e[key]
		elif _emotion == 'min_max/all':
			score = e_score.min_max_score(tweet, _wiktionary=False)
			e = get_emotion(score)
			for key in e:
				feats[key] = e[key]

		if description:
			d = get_desc(tweet_preprocessing(description[index]))
			for key in d:
				feats[key] = d[key]

		X.append(feats)
	return X

def get_postags(tokens, pos_tagger, flag):
	#Setup that tbh not sure why we need it but it's in train_tagger
	for j in range(pos_tagger.window):
		tokens.insert(0, pos_tagger.START_SYMBOL+'\\'+pos_tagger.START_SYMBOL)
		tokens.append(pos_tagger.STOP_SYMBOL+'\\'+pos_tagger.STOP_SYMBOL)
	for j in range(len(tokens)):
		tokens[j] = list(tokens[j].split('\\'))

	result = {}
	tags = []

	tagged_tweet = pos_tagger.tag_sents([tokens,], 'tweet')[0].replace('\n', '')
	for t in tagged_tweet.split(): tags.append(t.split('\\')[1])
	if flag == 'u':
		counts = dict()
		for i in tags: counts[i] = counts.get(i, 0) + 1
		return counts
	if flag == 'b':
		return get_bigrams(tags)


'''def read_in_postags(infile):
    result = {}
    count = 0
    f = open(infile, 'rU')
    for line in f.readlines():
        result[count] = (line)
        count += 1
    f.close()
    return result'''

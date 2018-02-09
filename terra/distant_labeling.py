#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
distant Labeling Class for Self-training on gang_intervention project
By: Terra
'''

import re
import math
import operator
import sys
sys.path.append('code/tools') 

from tools import tweet_preprocessing, get_label_sets

class DistantLabels:
	def importance(self, tweets, num_indicators):
	    no_fly = ['�e_', '“USER_HANDLE:'] #things we don't want to use as indicators

	    aggress, loss = get_label_sets()
	    aggress_tf, loss_tf, other_tf = {}, {}, {}
	    aggress_df, loss_df, other_df = {}, {}, {}
	    vocab = set()

	    #get vocab
	    for t in tweets:
	            vocab.update(tweet_preprocessing(t[0]))

	    #initalize td, df scores
	    for v in vocab:
	            aggress_tf[v], loss_tf[v], other_tf[v] = 0.0, 0.0, 0.0
	            aggress_df[v], loss_df[v], other_df[v] = 0.0, 0.0, 0.0

	    #get term, doc counts for vocab on each label
	    for tweet in tweets:
	            found = False
	            for l in aggress:
	                    if l in tweet[1].lower() and found == False:
	                            found = True
	                            aggress_tf = self.add_term_frequencies(tweet[0], aggress_tf)
	                            aggress_df = self.add_doc_frequencies(tweet[0], aggress_df)
	            for l in loss:
	                    if l in tweet[1].lower() and found == False:
	                            found = True
	                            loss_tf = self.add_term_frequencies(tweet[0], loss_tf)
	                            loss_df = self.add_doc_frequencies(tweet[0], loss_df)
	            if found == False:
	                    other_tf = self.add_term_frequencies(tweet[0], other_tf)
	                    other_df = self.add_doc_frequencies(tweet[0], other_df)

	    #normalization counts
	    aggress_total_tf = sum([aggress_tf[v] for v in vocab])
	    loss_total_tf = sum([loss_tf[v] for v in vocab])
	    other_total_tf = sum([other_tf[v] for v in vocab])

	    aggress_total_df =  sum([aggress_df[v] for v in vocab])
	    if aggress_total_df == 0: aggress_total_df = 1
	    loss_total_df =  sum([loss_df[v] for v in vocab])
	    if loss_total_df == 0: loss_total_df = 1
	    other_total_df =  sum([other_df[v] for v in vocab])
	    if other_total_df == 0: other_total_df = 1

	    #caluclate aggress, loss, other importance score: (tf_pos. * log(df_pos.)) - (tf_neg. * log(df_neg.))

	    aggress_scores, loss_scores, other_scores = {}, {}, {}
	    for v in vocab:
	            #aggress importance score
	            aggress_scores[v] = (aggress_tf[v]/aggress_total_tf)*(aggress_df[v]/aggress_total_df)
	            aggress_scores[v] -= ((loss_tf[v]+other_tf[v])/(loss_total_tf+other_total_tf))*((loss_df[v]+other_df[v])/(loss_total_df+other_total_df))

	            #loss importance score
	            loss_scores[v] = (loss_tf[v]/loss_total_tf)*(loss_df[v]/loss_total_df)
	            loss_scores[v] -= ((aggress_tf[v]+other_tf[v])/(aggress_total_tf+other_total_tf))*((aggress_df[v]+other_df[v])/(aggress_total_df+other_total_df))

	            #other importance score
	            other_scores[v] = (other_tf[v]/other_total_tf)*(other_df[v]/other_total_df)
	            other_scores[v] -= ((loss_tf[v]+aggress_tf[v])/(loss_total_tf+aggress_total_tf))*((loss_df[v]+aggress_df[v])/(loss_total_df+aggress_total_df))

	    #choose indicators based on importance score
	    sorted_aggress = [k[0] for k in sorted(aggress_scores.items(), key=operator.itemgetter(1), reverse=True) if k[0] not in no_fly]
	    sorted_loss = [k[0] for k in sorted(loss_scores.items(), key=operator.itemgetter(1), reverse=True) if k[0] not in no_fly]
	    sorted_other = [k[0] for k in sorted(other_scores.items(), key=operator.itemgetter(1), reverse=True) if k[0] not in no_fly]

	    sorted_aggress, sorted_loss, sorted_other = self.no_conficting_indicators(num_indicators, sorted_aggress, sorted_loss, sorted_other)

	    for s in sorted_aggress[:num_indicators]: print 'aggress '+str(s) #TESTING
	    for s in sorted_loss[:num_indicators]: print 'loss '+str(s) #TESTING
	    for s in sorted_other[:num_indicators]: print 'other '+str(s) #TESTING

	    return sorted_aggress[:num_indicators], sorted_loss[:num_indicators], sorted_other[:num_indicators]

	#tweets passed in as array of (content, label) pairs
	def indicators_with_tfidf(self, tweets, num_indicators):
		no_fly = ['�e_', '“USER_HANDLE:'] #things we don't want to use as indicators

		aggress, loss = get_label_sets()
		aggress_counts, loss_counts, other_counts, = {}, {}, {} 
		
		#get counts for each label
		for tweet in tweets:
			found = False
			for l in aggress:
				if l in tweet[1].lower() and found == False:
					found = True
					aggress_counts = self.add_term_frequencies(tweet[0], aggress_counts)
			for l in loss:
				if l in tweet[1].lower() and found == False:
					found = True
					loss_counts = self.add_term_frequencies(tweet[0], loss_counts)
			if found == False: 
				other_counts = self.add_term_frequencies(tweet[0], other_counts)

		#get tf-idf scores for each of the 3 groups
		aggress_tfidf, loss_tfidf, other_tfidf = {}, {}, {} 
		for key in aggress_counts.keys():
			doc_count = 1
			if key in loss_counts.keys(): doc_count += 1
			if key in other_counts.keys(): doc_count += 1
			aggress_tfidf[key] = aggress_counts[key] * (math.log(3/float(doc_count)))

		for key in loss_counts.keys():
			doc_count = 1
			if key in aggress_counts.keys(): doc_count += 1
			if key in other_counts.keys(): doc_count += 1
			loss_tfidf[key] = loss_counts[key] * (math.log(3/float(doc_count)))

		for key in other_counts.keys():
			doc_count = 1
			if key in aggress_counts.keys(): doc_count += 1
			if key in loss_counts.keys(): doc_count += 1
			other_tfidf[key] = other_counts[key] * (math.log(3/float(doc_count)))

		#choose indicators based on tf-idf
		sorted_aggress = [k[0] for k in sorted(aggress_tfidf.items(), key=operator.itemgetter(1), reverse=True) if k[0] not in no_fly]
		sorted_loss = [k[0] for k in sorted(loss_tfidf.items(), key=operator.itemgetter(1), reverse=True) if k[0] not in no_fly]
		sorted_other = [k[0] for k in sorted(other_tfidf.items(), key=operator.itemgetter(1), reverse=True) if k[0] not in no_fly]

#		if _only_emojis:
#			try:
#			# UCS-4
#				patt = re.compile(u'([\U00002600-\U000027BF])|([\U0001f300-\U0001f64F])|([\U0001f680-\U0001f6FF])')
#			except re.error:
#			# UCS-2
#				patt = re.compile(u'([\u2600-\u27BF])|([\uD83C][\uDF00-\uDFFF])|([\uD83D][\uDC00-\uDE4F])|([\uD83D][\uDE80-\uDEFF])')
#			sorted_aggress = [s for s in sorted_aggress if patt.match(s.decode('utf-8'))]
#			sorted_loss = [s for s in sorted_loss if patt.match(s.decode('utf-8'))]
#			sorted_other = [s for s in sorted_other if patt.match(s.decode('utf-8'))]

		sorted_aggress, sorted_loss, sorted_other = self.no_conficting_indicators(num_indicators, sorted_aggress, sorted_loss, sorted_other)

		for s in sorted_aggress[:num_indicators]: print 'aggress '+str(s) #TESTING
		for s in sorted_loss[:num_indicators]: print 'loss '+str(s) #TESTING
		for s in sorted_other[:num_indicators]: print 'other '+str(s) #TESTING

		return sorted_aggress[:num_indicators], sorted_loss[:num_indicators], sorted_other[:num_indicators]	

	def add_term_frequencies(self, tweet, counts):
		tokens = tweet_preprocessing(tweet)
		for t in tokens:
			try: 
				counts[t] = counts[t]+1
			except:
				counts[t] = 1
		return counts

	def add_doc_frequencies(self, tweet, counts):
		already_seen = []
		tokens = tweet_preprocessing(tweet)
		for t in tokens:
			if t not in already_seen:
				try: 
					counts[t] = counts[t]+1
				except:
					counts[t] = 1
				already_seen.append(t)
		return counts				

	def no_conficting_indicators(self, num_indicators, sorted_aggress, sorted_loss, sorted_other):
		i = 0
		while i < num_indicators:
			a = sorted_aggress[i]
			l = sorted_loss[i]
			o = sorted_other[i]
			if a == l or a == o: 
				sorted_aggress.remove(a)
				if a in sorted_loss: sorted_loss.remove(a)
				if a in sorted_other: sorted_other.remove(a)
			elif l == o:
				if l in sorted_aggress: sorted_aggress.remove(l)
				sorted_loss.remove(l)
				sorted_other.remove(l)
			else: 
				if a in sorted_loss: sorted_loss.remove(a)
				if a in sorted_other: sorted_other.remove(a)
				if l in sorted_aggress: sorted_aggress.remove(l)
				if l in sorted_other: sorted_other.remove(l)	
				if o in sorted_aggress: sorted_aggress.remove(o)
				if o in sorted_loss: sorted_loss.remove(o)	
				i += 1	
		return sorted_aggress, sorted_loss, sorted_other


	def predict_distant_label(self, tweet):
		tokens = tweet_preprocessing(tweet)
		aggress_likelihood = len([t for t in tokens if t in self.aggress_indicators])
		loss_likelihood = len([t for t in tokens if t in self.loss_indicators])
		other_likelihood = len([t for t in tokens if t in self.other_indicators])

		if aggress_likelihood > 0 and aggress_likelihood >= loss_likelihood and aggress_likelihood >= other_likelihood:
			return ' '.join([t for t in tokens if t not in self.aggress_indicators]), 'aggress'
		elif loss_likelihood > 0 and loss_likelihood >= other_likelihood: 
			return ' '.join([t for t in tokens if t not in self.loss_indicators]), 'loss'
		elif other_likelihood > 0:
			return ' '.join([t for t in tokens if t not in self.other_indicators]), 'other'
		else:
			return None


	def __init__(self, seed_tweets, num_indicators=None, _features=False, tfidf=True):
		if not num_indicators: num_indicators = 5
		self._features = _features
		if tfidf:
			self.aggress_indicators, self.loss_indicators, self.other_indicators = self.indicators_with_tfidf(seed_tweets, num_indicators)
		else: 
			self.aggress_indicators, self.loss_indicators, self.other_indicators = self.importance(seed_tweets, num_indicators)
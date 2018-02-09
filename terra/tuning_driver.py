'''
Driver code for the automated optimization of tweet classifier
Author: Terra
'''
import sys
sys.path.append('rjk2147/src')
sys.path.append('code/tools')

import csv
from classification import *
import tagger as t
from features import train_tagger

#note: feats = 0: unigrams, 1: bigrams, 2: pos_tag feats, 3: description, 4: emotion feats, 5: k (num_feats)
#note: results = 0: loss precision, 1: loss recall, 2: loss f1, 3: aggress precision, 4: aggress recall, 5: agress f1
def output(writer, results, params, feats):
	feats_str = ''
	if feats[0]: feats_str += 'unigrams ; '
	if feats[1]: feats_str += 'bigrams ; '
	if feats[2] == 'u': feats_str += 'POS tags (unigrams) ; '
	if feats[2] == 'b': feats_str += 'POS tags (bigrams) ; '
	if feats[3]: feats_str += 'description ; '
	if len(feats[4]) > 0: feats_str += 'emotion = '+feats[4]+' ; '

	d = {'params': params, 'feats': feats_str , 'num_feats': feats[5], 'f1 (sought)': results[2], 'precision (sought)': results[0], 'recall (sought)': results[1], 'f1 (nsought)': results[5], 'precision (nsought)': results[3], 'recall (nsought)': results[4]}
	print d #TESTING
	writer.writerow(d)
	return

which_model = sys.argv[1]
label = sys.argv[2]

class_weight = 'balanced' #NOT passed in
train_file = 'data/classification/_train/train_full.csv'
dev_file = 'data/classification/_dev/dev_full.csv'

#svm params: C, loss
svm_params = [range(2, 6), ['hinge', 'squared_hinge']]

#preceptron params: n_iter
perceptron_params = [range(3, 7)]

#logistic regression params: C, solver
log_reg_params = [range(3, 8), ['liblinear']]

#feature settings: unigrams, bigrams, POS unigrams, POS bigrams, description, avg emotion scores, avg ee only, avg norm only, dev emotion, min_max emotion
unigram_feats = [True]
bigram_feats = [True, False]
POS_feats = ['u', 'b', 'n']
#POS_feats = ['n']
#description = [True, False]
description = [False]
emotion = ['avg/all', 'dev/all', 'min_max/all', 'none']

num_feats = range(5, 16)

if which_model == 'svm':
	f_name = 'results/model_optimization/svm_full_'+label+'.csv'
	out_f = open(f_name, 'wb') #continuing setup
	writer = csv.DictWriter(out_f, ['params', 'feats', 'num_feats', 'f1 (sought)', 'precision (sought)', 'recall (sought)', 'f1 (nsought)', 'precision (nsought)', 'recall (nsought)'])
	writer.writeheader()

	pos_tagger = train_tagger()

	model = 'svm'
	for C in svm_params[0]:
		C = C*.1
		for loss in svm_params[1]:
			params = 'C = '+str(C)+' ; loss = '+loss
			print params #TESTING

			for u in unigram_feats:
				for b in bigram_feats:
					for s in POS_feats:
						for d in description:
							for e in emotion:
								for k in num_feats:
									k = k*100
									feats = [u, b, s, d, e, k]
									print feats #TESTING

									results = classify(train_file, dev_file, model, label, feats=feats, C=C, svm_loss=loss, pos_tagger=pos_tagger)
									output(writer, results, params, feats) 

	out_f.close()

if which_model == 'perceptron':
	f_name = 'results/model_optimization/perceptron_full_'+label+'.csv'
	out_f = open(f_name, 'wb')
	writer = csv.DictWriter(out_f, ['params', 'feats', 'num_feats', 'f1 (sought)', 'precision (sought)', 'recall (sought)', 'f1 (nsought)', 'precision (nsought)', 'recall (nsought)'])
	writer.writeheader()

	pos_tagger = train_tagger()

	model = 'perceptron'
	for n in perceptron_params[0]:
		params = 'num_iter = '+str(n)
		print params #TESTING

		for u in unigram_feats:
				for b in bigram_feats:
					for s in POS_feats:
						for d in description:
							for e in emotion:
								for k in num_feats:
									k = k*100
									feats = [u, b, s, d, e, k]
									print feats #TESTING

									results = classify(train_file, dev_file, model, label, feats=feats, n=n, pos_tagger=pos_tagger)
									output(writer, results, params, feats) 

	out_f.close()


if which_model == 'log_reg':
	f_name = 'results/model_optimization/log_regression_full_'+label+'.csv'
	out_f = open(f_name, 'wb')
	writer = csv.DictWriter(out_f, ['params', 'feats', 'num_feats', 'f1 (sought)', 'precision (sought)', 'recall (sought)', 'f1 (nsought)', 'precision (nsought)', 'recall (nsought)'])
	writer.writeheader()

	pos_tagger = train_tagger()

	model = 'log_reg'
	for C in log_reg_params[0]:
		C = C*.1
		for solver in log_reg_params[1]:
			params = 'C = '+str(C)+' ; '+solver
			print params #TESTING

			for u in unigram_feats:
				for b in bigram_feats:
					for s in POS_feats:
						for d in description:
							for e in emotion:
								for k in num_feats:
									k = k*100
									feats = [u, b, s, d, e, k]
									print feats #TESTING

									results = classify(train_file, dev_file, model, label, feats=feats, C=C, pos_tagger=pos_tagger)
									output(writer, results, params, feats) 
			
	out_f.close()
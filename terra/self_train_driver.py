import sys
sys.path.append('code/tools')
sys.path.append('code/labeling')

from tools import tweet_preprocessing, read_in_data, write_out_data, print_stats
from features import train_tagger
from distant_labeling import DistantLabels
from classification import classify

def semi_supervised_learning(train_file, unlabeled_file, eval_file, num_indicators, tfidf):
	seed_tweets = read_in_data(train_file)
	unlabeled_tweets = [t[0] for t in read_in_data(unlabeled_file)]

	#defaults for optional parameters 
	nl = DistantLabels(seed_tweets, num_indicators=num_indicators, tfidf=tfidf)
	distant_tweets = []
	for t in unlabeled_tweets:
		prediction = nl.predict_distant_label(t)
		if prediction: distant_tweets.append(prediction)
	write_out_data(distant_tweets, eval_file)

	#testing
	aggress_count = len([a for a in distant_tweets if a[1] =='aggress'])
	loss_count = len([a for a in distant_tweets if a[1] =='loss'])
	other_count = len([a for a in distant_tweets if a[1] =='other'])

	print 'num distant aggress = '+str(aggress_count)
	print 'num distant loss = '+str(loss_count)
	print 'num distant other = '+str(other_count)
	print 'num distant total = '+str(aggress_count+loss_count+other_count)
	print 'num before distant labeling = '+str(len(unlabeled_tweets))
	#\end testing

	num_updated = 1
	while num_updated > 0:
		num_updated = 0
		results = classify(train_file, eval_file, model, label, feats=feats, pos_tagger=pos_tagger, C=C, svm_loss=loss)
		eval_tweets = read_in_data(eval_file)
		verified_tweets = read_in_data(train_file)
		predictions = results[-1]

		for i in range(0, len(predictions)):
			if predictions[i] == 1 and eval_tweets[i][1] == label: 
				verified_tweets.append(eval_tweets[i])
				eval_tweets[i] = ''
				num_updated += 1
			elif predictions[i] == 0 and eval_tweets[i][1] != label: 
				verified_tweets.append(eval_tweets[i])
				eval_tweets[i] = ''
				num_updated += 1
		eval_tweets = [e for e in eval_tweets if e != '']

		write_out_data(eval_tweets, eval_file)
		write_out_data(verified_tweets, verified_file)
		train_file = verified_file

	return train_file


'''
MAIN METHOD
'''

#input files
train_file = 'data/classification/_train/train_full.csv'
unlabeled_file = 'data/top_ten/top_ten_communicators_noRTs.csv'

#temporary files for processing
rejected_file = 'results/semi_supervised/rejected_tweets.csv'
eval_file = 'results/semi_supervised/distant_labels.csv'
verified_file = 'results/semi_supervised/verified_labels.csv'

#file to test system on
#dev_file = 'data/_dev/dev_full.csv'
dev_file = 'data/arrogant_bubba.csv'

num_indicators = 3

#set up classification
model = 'svm'
label = 'loss_aggress'
feats = [1, 1, 'b', 0, 'n', 800]
C=0.5
loss = 'hinge'

pos_tagger = None
if feats[2] == 'u' or feats[2] == 'b':
	pos_tagger = train_tagger()


#Fully-supervised classifier + results
sought_p, sought_r, sought_f1, nsought_p, nsought_r, nsought_f1, predictions = classify(train_file, dev_file, model, label, feats, pos_tagger=pos_tagger, C=C, svm_loss=loss)
print_stats(sought_p, sought_r, sought_f1, nsought_p, nsought_r, nsought_f1, 'fully supervised classifier')


#Semi-supervised classifier with tf-idf + results
train_on_this = semi_supervised_learning(train_file, unlabeled_file, eval_file, num_indicators, True)
sought_p, sought_r, sought_f1, nsought_p, nsought_r, nsought_f1, predictions = classify(train_on_this, dev_file, model, label, feats=feats, pos_tagger=pos_tagger, C=C, svm_loss=loss)
print_stats(sought_p, sought_r, sought_f1, nsought_p, nsought_r, nsought_f1, 'semi-supervised (tf-idf) classifier')

#Semi-supervised classifier with "importance" metric + results
train_on_this = semi_supervised_learning(train_file, unlabeled_file, eval_file, num_indicators, False)
sought_p, sought_r, sought_f1, nsought_p, nsought_r, nsought_f1, predictions = classify(train_on_this, dev_file, model, label, feats=feats, pos_tagger=pos_tagger, C=C, svm_loss=loss)
print_stats(sought_p, sought_r, sought_f1, nsought_p, nsought_r, nsought_f1, 'semi-supervised (importance) classifier')
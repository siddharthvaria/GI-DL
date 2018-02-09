from _classification import classify
import sys
sys.path.append('code/tools')
from tools import *
from features import train_tagger

full_train_data = 'data/classification/_train/train_full.csv'
loss_aggress_train_data = 'data/classification/_train/train_aggress_loss.csv'
pred_dev_data = 'results/predictions/dev_predictions.csv'
default_dev_data = 'data/classification/_dev/dev_full.csv'
model1, model2 = 'svm', 'svm'

def cascade(tweet_file, technique=None):
    technique = ''
    dev_data = tweet_file
    labels_1 = 'loss_aggress'
    labels_2 = 'aggress'

    k1 = 1300
    k2 = 500
    feats_1 = [1, 1, 'n', 0, 'min_max/all', k1]
    feats_2 = [1, 1, 'u', 0, 'min_max/all', k2]
    C1 = 0.2
    C2 = 0.5
    loss1 = 'hinge'
    loss2 = 'squared_hinge'
    pos_tagger = train_tagger()

    all_test_tweets = read_in_data(dev_data)
    results = [-1 for i in range(0,len(all_test_tweets))]

    sought_p, sought_r, sought_f1, nsought_p, nsought_r, nsought_f1, predictions = classify(full_train_data, dev_data, model1, labels_1, feats_1, pos_tagger=pos_tagger, C=C1, svm_loss=loss1)
    print_stats(sought_p, sought_r, sought_f1, nsought_p, nsought_r, nsought_f1, 'first classifier')

    #save predictions to file
    if technique == 'loss_then_aggress':
    	predicted_test_tweets = []
    	for i in range(0, len(predictions)):
    		if predictions[i] == 0: predicted_test_tweets.append(all_test_tweets[i])
    	write_out_data(predicted_test_tweets, pred_dev_data)
    else:
    	predicted_test_tweets = []
    	for i in range(0, len(predictions)):
    		if predictions[i] == 1: predicted_test_tweets.append(all_test_tweets[i])
    		else: results[i] = 0
    	write_out_data(predicted_test_tweets, pred_dev_data)

        if len(predicted_test_tweets) == 0: return results

    sought_p, sought_r, sought_f1, nsought_p, nsought_r, nsought_f1, predictions = classify(loss_aggress_train_data, pred_dev_data, model2, labels_2, feats_2, pos_tagger=pos_tagger, C=C2, svm_loss=loss2)
    print_stats(sought_p, sought_r, sought_f1, nsought_p, nsought_r, nsought_f1, 'second classifier')

    #get the predicted tweets and return them
    r = 0
    for p in predictions:
    	while results[r] == 0: r += 1
    	if p == 1: results[r] = 1
    	else: results[r] = 0
    	r += 1
    	
    return results


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

def main():
    if len(sys.argv) > 1: 
        technique = sys.argv[1]
    else: technique = ''

    if len(sys.argv) > 2:
        dev_data = sys.argv[2]
    else: 
        dev_data = default_dev_data

    pos_tagger = train_tagger()

    print 'dev_data file: '+dev_data

    #choose label sets, features based on how we want to cascade the classifiers
    #either by "grouping" (weeding out gen first and then seperating loss and aggress)
    #or by "smallest" (getting aggression first, then loss)

    #feats = [_unigrams, _bigrams, _postags, _description, emotion]
    aggress, loss = get_label_sets()

    if technique == 'both_then_aggress':
        labels_1 = loss + aggress
        labels_2 = aggress

        k1 = 600
        k2 = 1300
        feats_1 = [1, 0, 'u', 0, 'min_max/all', k1]
        feats_2 = [1, 1, 'n', 0, 'min_max/all', k2]
        C1 = 0.5
        C2 = 0.3
        loss1 = 'squared_hinge'
        loss2 = 'hinge'

    elif technique == 'both_then_loss':
        labels_1 = loss + aggress
        labels_2 = loss

        k1 = 600
        k2 = 1100
        feats_1 = [1, 0, 'u', 0, 'min_max/all', k1]
        feats_2 = [1, 1, 'u', 0, 'min_max/all', k2]
        C1 = 0.5
        C2 = 0.3
        loss1 = 'squared_hinge'
        loss2 = 'hinge'

    elif technique == 'loss_then_aggress':
        labels_1 = loss
        labels_2 = aggress
        k1 = 1400
        k2 = 1200
        feats_1 = [1, 0, 'n', 0, 'min_max/all', k1]
        feats_2 = [1, 0, 'u', 0, 'min_max/all', k2]
        C1 = 0.3
        C2 = 0.4
        loss1 = 'hinge'
        loss2 = 'hinge'

    #if technique == 'precision_aggress':
    else:
        labels_1 = loss + aggress
        labels_2 = aggress

        k1 = 1300
        k2 = 500
        feats_1 = [1, 1, 'n', 0, 'min_max/all', k1]
        feats_2 = [1, 1, 'u', 0, 'min_max/all', k2]
        C1 = 0.2
        C2 = 0.5
        loss1 = 'hinge'
        loss2 = 'squared_hinge'

    all_test_tweets = read_in_data(dev_data)
    results = [-1 for i in range(0,len(all_test_tweets))]

    sought_p, sought_r, sought_f1, nsought_p, nsought_r, nsought_f1, predictions = classify(full_train_data, dev_data, model1, labels_1, feats_1, pos_tagger=pos_tagger, C=C1, svm_loss=loss1)
    print_stats(sought_p, sought_r, sought_f1, nsought_p, nsought_r, nsought_f1, 'first classifier')

    #save predictions to file
    if technique == 'loss_then_aggress':
        predicted_test_tweets = []
        for i in range(0, len(predictions)):
            if predictions[i] == 0: predicted_test_tweets.append(all_test_tweets[i])
            else: results[i] = 0
        write_out_data(predicted_test_tweets, pred_dev_data)
    else:
        predicted_test_tweets = []
        for i in range(0, len(predictions)):
            if predictions[i] == 1: predicted_test_tweets.append(all_test_tweets[i])
            else: results[i] = 0
        write_out_data(predicted_test_tweets, pred_dev_data)

    if len(predicted_test_tweets) == 0: return results

    sought_p, sought_r, sought_f1, nsought_p, nsought_r, nsought_f1, predictions = classify(loss_aggress_train_data, pred_dev_data, model2, labels_2, feats_2, pos_tagger=pos_tagger, C=C2, svm_loss=loss2)
    print_stats(sought_p, sought_r, sought_f1, nsought_p, nsought_r, nsought_f1, 'second classifier')

    r = 0
    for p in predictions:
        while results[r] == 0: r += 1
        if p == 1: results[r] = 1
        else: results[r] = 0
        r += 1
    print results


if __name__ == "__main__":
        main()

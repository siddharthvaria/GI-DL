import scipy
from features import train_tagger
import sys
sys.path.append('code/classification')
from classification import classify
from tools import *

def write_out_predictions(pred, filename):
	output = open(filename, 'w')
	for p in pred:
		output.write(str(p)+'\n')
	output.close()

#if using outside package: java de/pado/sigf/AverageART ../sig_test/CC_aggress_a ../sig_test/CC_aggress_b 100

pos_tagger = train_tagger()



model1 = 'svm'
model2 = 'svm'

full_train_data = 'data/_train/train_full.csv'
loss_aggress_train_data = 'data/_train/train_aggress_loss.csv'
pred_dev_data = 'results/predictions/dev_sig_predictions.csv'
dev_data = 'data/_test/test_full.csv'

#CC - aggression
labels_1 = 'loss_aggress'
labels_2 = 'aggress'

k1 = 800
k2 = 1500
feats_1 = [1, 1, 'b', 0, 'none', k1]
feats_2 = [1, 1, 'n', 0, 'min_max/all', k2]
C1 = 0.5
C2 = 0.3
loss1 = 'hinge'
loss2 = 'hinge'

sought_p, sought_r, sought_f1, nsought_p, nsought_r, nsought_f1, predictions = classify(full_train_data, dev_data, model1, labels_1, feats_1, pos_tagger=pos_tagger, C=C1, svm_loss=loss1)

#save predictions to file
all_test_tweets = read_in_data(dev_data)
predicted_test_tweets = []
for i in range(0, len(predictions)):
	if predictions[i] == 1: predicted_test_tweets.append(all_test_tweets[i])
write_out_data(predicted_test_tweets, pred_dev_data)

sought_p, sought_r, sought_f1, nsought_p, nsought_r, nsought_f1, predictions2 = classify(loss_aggress_train_data, pred_dev_data, model2, labels_2, feats_2, pos_tagger=pos_tagger, C=C2, svm_loss=loss2)

a = []
c = 0
for i in range(0, len(predictions)):
	if predictions[i] == 1: 
		a.append(predictions2[c])
		c += 1
	else: a.append(0)

#BCF - aggression (unigram model)
train_file = 'data/_train/train_full.csv'
dev_file = 'data/_test/test_full.csv'
model = 'svm'
label = 'aggress'
feats = [1, 0, 'n', 0, 'n', 1500]
C=0.4
loss = 'hinge'

results = classify(train_file, dev_file, model, label, feats=feats, pos_tagger=pos_tagger, C=C, svm_loss=loss)
b = results[-1]

print len(a) #TESTING
print len(b) #TESTING
print scipy.stats.ttest_ind(a, b)
print 'p value of CC aggression = '+str(scipy.stats.ttest_ind(a, b)[1])
print

write_out_predictions(a, 'results/significance_testing/CC_aggress_a')
write_out_predictions(b, 'results/significance_testing/CC_aggress_b')




#CC - loss
labels_1 = 'loss_aggress'
labels_2 = 'loss'

k1 = 800
k2 = 1000
feats_1 = [1, 1, 'b', 0, 'none', k1]
feats_2 = [1, 0, 'u', 0, 'min_max/all', k2]
C1 = 0.5
C2 = 0.3
loss1 = 'hinge'
loss2 = 'hinge'

sought_p, sought_r, sought_f1, nsought_p, nsought_r, nsought_f1, predictions = classify(full_train_data, dev_data, model1, labels_1, feats_1, pos_tagger=pos_tagger, C=C1, svm_loss=loss1)

#save predictions to file
all_test_tweets = read_in_data(dev_data)
predicted_test_tweets = []
for i in range(0, len(predictions)):
	if predictions[i] == 1: predicted_test_tweets.append(all_test_tweets[i])
write_out_data(predicted_test_tweets, pred_dev_data)

sought_p, sought_r, sought_f1, nsought_p, nsought_r, nsought_f1, predictions2 = classify(loss_aggress_train_data, pred_dev_data, model2, labels_2, feats_2, pos_tagger=pos_tagger, C=C2, svm_loss=loss2)

a = []
c = 0
for i in range(0, len(predictions)):
	if predictions[i] == 1: 
		a.append(predictions2[c])
		c += 1
	else: a.append(0)

#BCF - loss (unigram model)
train_file = 'data/_train/train_full.csv'
dev_file = 'data/_test/test_full.csv'
model = 'svm'
label = 'loss'
feats = [1, 0, 'n', 0, 'none', 700] 
C=0.4
loss = 'squared_hinge'

results = classify(train_file, dev_file, model, label, feats=feats, pos_tagger=pos_tagger, C=C, svm_loss=loss)
b = results[-1]

print len(a) #TESTING
print len(b) #TESTING
print scipy.stats.ttest_ind(a, b)
print 'p value of CC loss = '+str(scipy.stats.ttest_ind(a, b)[1])
print

write_out_predictions(a, 'results/significance_testing/CC_loss_a')
write_out_predictions(b, 'results/significance_testing/CC_loss_b')




#BCF - aggression (full model)
train_file = 'data/_train/train_full.csv'
dev_file = 'data/_test/test_full.csv'
model = 'svm'
label = 'aggress'
feats = [1, 1, 'u', 0, 'min_max/all', 1500]
C=0.4
loss = 'hinge'

results = classify(train_file, dev_file, model, label, feats=feats, pos_tagger=pos_tagger, C=C, svm_loss=loss)
a = results[-1]

#BCF - aggression (unigram model)
train_file = 'data/_train/train_full.csv'
dev_file = 'data/_test/test_full.csv'
model = 'svm'
label = 'aggress'
feats = [1, 0, 'n', 0, 'n', 1500]
C=0.4
loss = 'hinge'

results = classify(train_file, dev_file, model, label, feats=feats, pos_tagger=pos_tagger, C=C, svm_loss=loss)
b = results[-1]

print len(a) #TESTING
print len(b) #TESTING
print 'p value of BCF aggression = '+str(scipy.stats.ttest_ind(a, b)[1])
print

write_out_predictions(a, 'results/significance_testing/BCF_aggress_a')
write_out_predictions(b, 'results/significance_testing/BCF_aggress_b')


#BCF - loss (full model)
train_file = 'data/_train/train_full.csv'
dev_file = 'data/_test/test_full.csv'
model = 'svm'
label = 'loss'
feats = [1, 0, 'b', 0, 'none', 700] 
C=0.4
loss = 'squared_hinge'

results = classify(train_file, dev_file, model, label, feats=feats, pos_tagger=pos_tagger, C=C, svm_loss=loss)
a = results[-1]

#BCF - loss (unigram model)
train_file = 'data/_train/train_full.csv'
dev_file = 'data/_test/test_full.csv'
model = 'svm'
label = 'loss'
feats = [1, 0, 'n', 0, 'none', 700] 
C=0.4
loss = 'squared_hinge'

results = classify(train_file, dev_file, model, label, feats=feats, pos_tagger=pos_tagger, C=C, svm_loss=loss)
b = results[-1]

print len(a) #TESTING
print len(b) #TESTING
print 'p value of BCF loss = '+str(scipy.stats.ttest_ind(a, b)[1])
print

write_out_predictions(a, 'results/significance_testing/BCF_loss_a')
write_out_predictions(b, 'results/significance_testing/BCF_loss_b')


#BCF - aggress+loss (full model)
train_file = 'data/_train/train_full.csv'
dev_file = 'data/_test/test_full.csv'
model = 'svm'
label = 'loss_aggress'
feats = [1, 1, 'b', 0, 'n', 800]
C=0.5
loss = 'hinge'

results = classify(train_file, dev_file, model, label, feats=feats, pos_tagger=pos_tagger, C=C, svm_loss=loss)
a = results[-1]

#BCF - aggress+loss (unigram model)
train_file = 'data/_train/train_full.csv'
dev_file = 'data/_test/test_full.csv'
model = 'svm'
label = 'loss_aggress'
feats = [1, 0, 'n', 0, 'n', 800]
C=0.5
loss = 'hinge'

results = classify(train_file, dev_file, model, label, feats=feats, pos_tagger=pos_tagger, C=C, svm_loss=loss)
b = results[-1]

print len(a) #TESTING
print len(b) #TESTING
print 'p value of BCF aggress+loss = '+str(scipy.stats.ttest_ind(a, b)[1])
print

write_out_predictions(a, 'results/significance_testing/BCF_loss_aggress_a')
write_out_predictions(b, 'results/significance_testing/BCF_loss_aggress_b')


#BCS - aggression (full model)
train_file = 'data/_train/train_aggress_loss.csv'
dev_file = 'data/_test/test_aggress_loss.csv'
model = 'svm'
label = 'aggress'
feats = [1, 1, 'n', 0, 'min_max/all', 1500]
C=0.3
loss = 'hinge'

results = classify(train_file, dev_file, model, label, feats=feats, pos_tagger=pos_tagger, C=C, svm_loss=loss)
a = results[-1]

#BCS - aggression (unigram model)
train_file = 'data/_train/train_aggress_loss.csv'
dev_file = 'data/_test/test_aggress_loss.csv'
model = 'svm'
label = 'aggress'
feats = [1, 0, 'n', 0, 'n', 1500]
C=0.3
loss = 'hinge'

results = classify(train_file, dev_file, model, label, feats=feats, pos_tagger=pos_tagger, C=C, svm_loss=loss)
b = results[-1]

print len(a) #TESTING
print len(b) #TESTING
print 'p value of BCS aggression = '+str(scipy.stats.ttest_ind(a, b)[1])
print

write_out_predictions(a, 'results/significance_testing/BCS_aggress_a')
write_out_predictions(b, 'results/significance_testing/BCS_aggress_b')


#BCS - loss (full model)
train_file = 'data/_train/train_aggress_loss.csv'
dev_file = 'data/_test/test_aggress_loss.csv'
model = 'svm'
label = 'loss'
feats = [1, 0, 'u', 0, 'min_max/all', 1000]
C=0.3
loss = 'hinge'

results = classify(train_file, dev_file, model, label, feats=feats, pos_tagger=pos_tagger, C=C, svm_loss=loss)
a = results[-1]

#BCS - loss (unigram model)
train_file = 'data/_train/train_aggress_loss.csv'
dev_file = 'data/_test/test_aggress_loss.csv'
model = 'svm'
label = 'loss'
feats = [1, 0, 'n', 0, 'n', 1000]
C=0.3
loss = 'hinge'

results = classify(train_file, dev_file, model, label, feats=feats, pos_tagger=pos_tagger, C=C, svm_loss=loss)
b = results[-1]

print len(a) #TESTING
print len(b) #TESTING
print 'p value of BCS loss = '+str(scipy.stats.ttest_ind(a, b)[1])
print

write_out_predictions(a, 'results/significance_testing/BCS_loss_a')
write_out_predictions(b, 'results/significance_testing/BCS_loss_b')
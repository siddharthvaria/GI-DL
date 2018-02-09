import pickle
import csv

from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_selection import SelectKBest

from tools import read_in_data, get_label_sets
from features import get_feats

import numpy as np

aggress, loss = get_label_sets()

feats = [1, 1, 'n', 0, 'min_max/all', 1300]

clf = pickle.load(open('trained_svm.pkl', 'r'))
selector = pickle.load(open('feature_selector.pkl', 'r'))

train_file = 'nov-new-dataset/train.csv'
test_file = 'nov-new-dataset/dev.csv'

tweets = read_in_data(train_file)
len_training = len(tweets)
tweets += read_in_data(test_file)

X_o = [i[0] for i in tweets]
X = get_feats(X_o, _unigrams=feats[0], _bigrams=feats[1], _postags=feats[2], pos_tagger=None, _emotion=feats[4], description=feats[3])

y = []

for t in tweets:
    if len([1 for s in loss if s in t[1]])>0: y.append(0)
    elif len([1 for s in aggress if s in t[1]])>0: y.append(2)
    else: y.append(1)

y_train = y[:len_training]
y_test = y[len_training:]

v = DictVectorizer()
X_vec = v.fit_transform(X).toarray()

X_train = X_vec[:len_training]
X_test = X_vec[len_training:]

feature_selector = SelectKBest(k=feats[5])
train_x = feature_selector.fit_transform(X_train, y_train)
train_x = selector.transform(X_train)

# train

y_pred = clf.predict(train_x)
decs = clf.decision_function(train_x)

print(np.shape(decs))

out = open('train_probs.tsv', 'w')

for i in range(len(y_pred)):
    out.write(str(decs[i][0]) + '\t' + str(decs[i][1]) + '\t' + str(decs[i][2]) + '\n')

# dev
test_x = selector.transform(X_test)
    
y_pred = clf.predict(test_x)
decs = clf.decision_function(test_x)

print(np.shape(decs))

out = open('dev_probs.tsv', 'w')

for i in range(len(y_pred)):
    out.write(str(decs[i][0]) + '\t' + str(decs[i][1]) + '\t' + str(decs[i][2]) + '\n')

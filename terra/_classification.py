'''
Classification of tweets for Gang Intervention Project
Author: Terra
'''
import sys
sys.path.append('code/tools')

from sklearn import svm
from sklearn.linear_model import Perceptron, LogisticRegression

from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import precision_score, recall_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.externals import joblib

import itertools
import csv 
import nltk
import operator
import numpy
import time

from tools import get_label_sets, fscore, read_in_data
from features import get_feats, train_tagger


def modeling(train_x , test_x, train_y, test_y, top_k, v, model=False, C=False, svm_loss=False, n=False):
    # Select top k features
    try:
        feature_selector = SelectKBest(chi2, k=top_k)
        train_x = feature_selector.fit_transform(train_x,train_y)
        test_x = feature_selector.transform(test_x)
        selected_features = [(v.get_feature_names()[i],feature_selector.scores_[i]) for i in feature_selector.get_support(indices=True)]
    except:
        feature_selector = SelectKBest(chi2, k='all')
        train_x = feature_selector.fit_transform(train_x,train_y)
        test_x = feature_selector.transform(test_x)
        selected_features = [(v.get_feature_names()[i],feature_selector.scores_[i]) for i in feature_selector.get_support(indices=True)]

#    if model == 'perceptron': clf = Perceptron(n_iter=n, class_weight='balanced')
#    elif model == 'log_reg': clf = LogisticRegression(C=C, solver='liblinear', class_weight='balanced')
#    else: clf = svm.LinearSVC(C=C, loss=svm_loss, class_weight='balanced')
    clf = svm.LinearSVC(C=C, loss=svm_loss, class_weight='balanced')

    clf.fit(train_x, train_y)
    y_pred = clf.predict(test_x)

    #get stats
    s = [1 for i in range(0, len(y_pred)) if y_pred[i] == test_y[i]]
    accuracy = float(len(s))/len(y_pred)

    threat_precision = precision_score(test_y, y_pred, pos_label=1)
    threat_recall = recall_score(test_y, y_pred, pos_label=1)

    nthreat_precision = precision_score(test_y, y_pred, pos_label=0)
    nthreat_recall = recall_score(test_y, y_pred, pos_label=0)

    return y_pred, threat_precision, threat_recall, nthreat_precision, nthreat_recall


#classic classification
def classify(train_file, test_file, model, label, feats, pos_tagger=None, C=False, svm_loss=False, n=False):
    reload(sys)
    sys.setdefaultencoding("utf-8")

    #start timer
    start_time = time.time()

    #get label sets
    aggress, loss = get_label_sets()

    #set params, features based on input variables
    if label == 'loss': sought_label = loss
    elif label == 'aggress': sought_label = aggress
    elif label == 'loss_aggress': sought_label= loss+aggress
    else: sought_label = label
    print sought_label
    class_weights = 'balanced'
    top_k = feats[5]

    _unigrams = feats[0]
    _bigrams = feats[1]

    _postags = feats[2]

    _description = feats[3]

    _emotion = feats[4]

    description = []
    tweets = []
    index = 0

    #load in training tweets
    tweets = read_in_data(train_file)
    len_training = len(tweets)
    #load in test tweets
    tweets = tweets + read_in_data(test_file)

    # get features
    X = [i[0] for i in tweets]
    X = get_feats(X, _unigrams=_unigrams, _bigrams=_bigrams, _postags=_postags, pos_tagger=pos_tagger, _emotion=_emotion, description=description)

    y = []
    for t in tweets:
        if len([1 for s in sought_label if s in t[1]])>0: y.append(1)
        else: y.append(0)
    y_train = y[:len_training]
    y_test = y[len_training:]
    print len(y_test)
    print len(y_train)

    #vectorize features
    v = DictVectorizer()
    X_vec = v.fit_transform(X)
    X_train = X_vec[:len_training]
    X_test = X_vec[len_training:]


    #train, score on SVM
    predictions, threat_p, threat_r, nthreat_p, nthreat_r = modeling(X_train, X_test, y_train, y_test, top_k, v, model=model, C=C, svm_loss=svm_loss, n=n)

    #output
    end_time = time.time()
    print 'time elapsed: '+str(end_time-start_time)+' seconds.'

    return threat_p, threat_r, fscore(threat_p, threat_r), nthreat_p, nthreat_r, fscore(nthreat_p, nthreat_r), predictions

#note: feats = 0: unigrams, 1: bigrams, 2: pos_tag feats, 3: description, 4: emotion feats, 5: k (num_feats)
#note: results = 0: loss f1, 1: loss precision, 2: loss recall, 3: aggress f1

def classify_with_existing_model(filepath, examples):
    reload(sys)
    sys.setdefaultencoding("utf-8") 

    #read in settings
    f = open(filepath+'settings.txt', 'r')
    label = f.readline().strip()
    feats = f.readline().strip().split(' ')
    f.close()

    #get label sets
    aggress, loss = get_label_sets()

    #set params, features based on input variables
    if label == 'loss': sought_label = loss
    elif label == 'aggress': sought_label = aggress
    elif label == 'loss_aggress': sought_label= loss+aggress
    else: sought_label = label
    print sought_label
    class_weights = 'balanced'
    top_k = feats[5]

    _unigrams = feats[0]
    _bigrams = feats[1]

    _postags = feats[2]
    if _postags:
        pos_tagger = train_tagger()
    else:
        pos_tagger = None

    _description = feats[3]

    _emotion = feats[4]

    description = []
    tweets = []
    index = 0

    X = get_feats(examples, _unigrams=_unigrams, _bigrams=_bigrams, _postags=_postags, pos_tagger=pos_tagger, _emotion=_emotion, description=description)

    #vectorize features
    v = DictVectorizer()
    X_test = v.fit_transform(X).todense().tolist()

    #score on exisiting SVM
    fs = joblib.load(filepath+'selection.pkl')
    clf = joblib.load(filepath+'model.pkl')
    X_test = fs.transform(X_test)
    predictions = clf.predict(X_test)
    #use log prob predict instead to get likelihood instead of binary classification?

    return predictions
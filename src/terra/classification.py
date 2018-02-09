'''
Classification of tweets for Gang Intervention Project
Author: Terra
'''
import sys
sys.path.append('code/tools')

from sklearn import svm
from sklearn.linear_model import Perceptron, LogisticRegression

from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.externals import joblib

from sklearn.grid_search import GridSearchCV

import itertools
import csv 
import nltk
import operator
import numpy
import time
import pickle

from tools import get_label_sets, fscore, read_in_data
from features import get_feats, train_tagger, get_wfeats

def score(clf, X, y):
    return f1_score(clf.predict(X), y, average='macro')

def modeling(train_x , test_x, train_y, test_y, top_k, v, model=False, C=False, svm_loss=False, n=False, verbose=True):
    x = train_x

    if verbose:
        print 'Generating feature selector and transforming features...'
    
    # Select top k features
    try:
        feature_selector = SelectKBest(k=top_k)#chi2, k=top_k)
        feature_selector.fit(train_x[-4194:], train_y[-4194:])
        train_x = feature_selector.transform(train_x,train_y)
        test_x = feature_selector.transform(test_x)
        selected_features = [(v.get_feature_names()[i],feature_selector.scores_[i]) for i in feature_selector.get_support(indices=True)]
    except:
        feature_selector = SelectKBest(k=top_k)#chi2, k='all')
        train_x = feature_selector.fit_transform(train_x,train_y)
        test_x = feature_selector.transform(test_x)
        selected_features = [(v.get_feature_names()[i],feature_selector.scores_[i]) for i in feature_selector.get_support(indices=True)]

    if verbose:
        print 'Fitting...'

#    if model == 'perceptron': clf = Perceptron(n_iter=n, class_weight='balanced')
#    elif model == 'log_reg': clf = LogisticRegression(C=C, solver='liblinear', class_weight='balanced')
#    else: clf = svm.LinearSVC(C=C, loss=svm_loss, class_weight='balanced')
    clf = svm.LinearSVC(C=C, loss=svm_loss, class_weight='balanced')

#    clf = GridSearchCV(svm.LinearSVC(loss=svm_loss, class_weight='balanced'), {'penalty': ['l2'], 'loss': ['hinge', 'squared_hinge'], 'C': [.0001, .0005, .001, .005, .01, .05, .1, .3, 1, 5, 10, 50, 100, 500]})

    clf.fit(train_x, train_y)

#    print clf.best_params_

    if verbose:
        print 'Predicting...'
        
    y_pred = clf.predict(test_x)

    if verbose:
        print 'Obtaining probabilities...'
        
    decs = clf.decision_function(test_x)

    if verbose:
        print 'Saving probabilities...'

    out = open('dev_probs.tsv', 'w')
    out.write('Loss\tOther\tAggression\n')

    for d in decs:
        out.write(str(d[0]) + '\t' + str(d[1]) + '\t' + str(d[2]) + '\n')

    pickle.dump(clf, open('trained_svm.pkl', 'w'))
    pickle.dump(feature_selector, open('feature_selector.pkl', 'w'))

    for i in range(len(y_pred)):
        pass
        #y_pred[i] = str(x[i]) + '\t' + str(y_pred[i])

    #get stats
    s = [1 for i in range(0, len(y_pred)) if y_pred[i] == test_y[i]]
    accuracy = float(len(s))/len(y_pred)

    preds = []

    for i in range(len(y_pred)):
        preds.append([test_x[i], y_pred[i], decs[i], test_y[i]])

    threat_precision = precision_score(test_y, y_pred, pos_label=0, average=None)[0]
    threat_recall = recall_score(test_y, y_pred, pos_label=0, average=None)[0]

    nthreat_precision = precision_score(test_y, y_pred, pos_label=1, average=None)[1]
    nthreat_recall = recall_score(test_y, y_pred, pos_label=1, average=None)[1]

    sthreat_precision = precision_score(test_y, y_pred, pos_label=2, average=None)[2]
    sthreat_recall = recall_score(test_y, y_pred, pos_label=2, average=None)[2]

    decs = clf.decision_function(train_x)

    out = open('train_probs.tsv', 'w')
    out.write('Loss\tOther\tAggression\n')

    for d in decs:
        out.write(str(d[0]) + '\t' + str(d[1]) + '\t' + str(d[2]) + '\n')

    return preds, threat_precision, threat_recall, nthreat_precision, nthreat_recall, sthreat_precision, sthreat_recall


#classic classification
def classify(train_file, test_file, model, label, feats, pos_tagger=None, C=False, svm_loss=False, n=False, verbose=True):
    reload(sys)
    sys.setdefaultencoding("utf-8")

    print 'in classify()'

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
    tweets = read_in_data(train_file)#[:2000]
    len_training = len(tweets)
    #load in test tweets
    tweets = tweets + read_in_data(test_file)

    # get features
    if verbose:
        print 'Obtaining features...'
    
    X_o = [i[0] for i in tweets]

    # TESTING PURPOSES ONLY
#    X_o = X_o [:100]
#    len_training = 80

    X_train = X_o[:len_training]
    X_test = X_o[len_training:]

#    print X_train
    
    X_train = get_feats(X_train, _unigrams=_unigrams, _bigrams=_bigrams, _postags=_postags, pos_tagger=pos_tagger, _emotion=_emotion, description=description)
#    wfeats = get_wfeats(X_o)

    #vectorize features
    if verbose:
        print 'Vectorizing...'

    v = DictVectorizer()

    X_train = v.fit_transform(X_train)#.toarray()

    print X_train.shape
    print X_train.getnnz()
#    exit(0)

#    print v.feature_names_

    X_test = get_feats(X_test, _unigrams=_unigrams, _bigrams=_bigrams, _postags=_postags, pos_tagger=pos_tagger, _emotion=_emotion, description=description, feature_names=v.feature_names_)
    
    X_test = v.transform(X_test)#.toarray()
    
    y = []
    for t in tweets:
# -------TERRA'S ORIGINAL CODE
#        if len([1 for s in sought_label if s in t[1]])>0: y.append(1)
#        else: y.append(0)
         if len([1 for s in loss if s in t[1]])>0: y.append(0)
         elif len([1 for s in aggress if s in t[1]])>0: y.append(2)
         else: y.append(1)
    y_train = y[:len_training]
    y_test = y[len_training:]
#    print len(y_test)
#    print len(y_train)

    # split
    if verbose:
        print 'Splitting...'

#    X_train = v.fit_transform(X_train).toarray()
#    X_test = v.transform(X_test).toarray()

    pickle.dump(v, open('terra_dictvectorizer.pkl', 'w'))

#    X_new = []
#
#    # add word embedding features
#    if verbose:
#        print 'Adding word embeddings...'
#    
#    for i in range(len(X_vec)):
#        x = X_vec[i]
##        print x
#        x = numpy.concatenate((x, wfeats[0][i]))
##        x = numpy.concatenate((x, wfeats[1][i]))
#        X_new.append(x)
#
#    #X_vec = numpy.array(X_new)
    
# Terra's original code
##    X_train = X_vec[:len_training] #X_new
#    X_test = X_vec[len_training:] #X_new
    
    #train, score on SVM
    predictions, threat_p, threat_r, nthreat_p, nthreat_r, sthreat_p, sthreat_r = modeling(X_train, X_test, y_train, y_test, top_k, v, model=model, C=C, svm_loss=svm_loss, n=n)

#    for i in range(len(predictions)):
#        print tweets[i] + ':\t' + str(predictions[i]) + '; ' + str(y_test[i])
    
    #output
    end_time = time.time()
    print 'time elapsed: '+str(end_time-start_time)+' seconds.'

    return threat_p, threat_r, fscore(threat_p, threat_r), nthreat_p, nthreat_r, fscore(nthreat_p, nthreat_r), sthreat_p, sthreat_r, fscore(sthreat_p, sthreat_r), predictions

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

    for i in range(len(predictions)):
        print tweets[i] + ':\t' + str(predictions[i]) + '; ' + str(y_test[i])

    return predictions

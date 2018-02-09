import sys
import nltk
import math
import time
import string
import numpy as np

import csv

import pybrain
from pybrain.datasets import supervised
from pybrain.tools.shortcuts import buildNetwork

from sklearn import svm
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.feature_selection import SelectKBest, chi2, f_classif

START_SYMBOL = '*'
STOP_SYMBOL = 'STOP'
RARE_SYMBOL = '_RARE_'
RARE_WORD_MAX_FREQ = 5
LOG_PROB_OF_ZERO = -1000

def count(to_count):
    l = sorted(to_count)
    counted = []
    i = 0
    while i < len(l):
        item = l[i]
        count = 1
        while i + count < len(l) and l[i+count] == item:
            count += 1
        counted.append((item, count))
        i += count
    return counted

def get_n_grams(tokens, n):
    n_grams = []
    for i in range(0, len(tokens)+1-n):
        n_gram = ()
        for j in range(0, n):
            n_gram = n_gram + (tokens[i+j],)
        n_grams.append(n_gram)
    return n_grams

def split_wordtags(train_sents):
    train_words = []
    train_tags = []
    for sentence in train_sents:
        tags = []
        words = []
        joined = sentence.split()
        joined.insert(0, START_SYMBOL+"\\"+START_SYMBOL)
        joined.insert(0, START_SYMBOL+"\\"+START_SYMBOL)
        joined.append(STOP_SYMBOL+"\\"+STOP_SYMBOL)
        for phrase in joined:
            for n in range(0, len(phrase)):
                i = len(phrase)-1 - n
                if (phrase[i] == '\\'):
                    words.append(phrase[0:i])
                    tags.append(phrase[i+1:])
                    break
        if tags == 'X':
            print
        train_words.append(words)
        train_tags.append(tags)

    return train_words, train_tags

def calc_trigrams(train_tags):
    unigram_tuples = []
    bigram_tuples = []
    trigram_tuples = []
    for sentence in train_tags:
        unigram_tuples.extend(sentence)
        bigram_tuples.extend(get_n_grams(sentence, 2))
        trigram_tuples.extend(get_n_grams(sentence, 3))

    unigram_tuples = [(unigram,) for unigram in unigram_tuples]

    unigram_count = count(unigram_tuples)
    bigram_count = count(bigram_tuples)
    trigram_count = count(trigram_tuples)
    all_sentences = math.log(float(len(train_tags)/float(len(unigram_tuples))), 2)

    unigram_p = {item[0]: math.log((float(item[1])/float(len(unigram_tuples))), 2) for item in unigram_count}

    bigram_p = {item[0]: math.log((float(item[1])/float(len(unigram_tuples))), 2)
                         - unigram_p.setdefault((item[0][0],),all_sentences) for item in bigram_count}
    trigram_p = {item[0]: math.log((float(item[1])/float(len(unigram_tuples))), 2)
                          - bigram_p.setdefault((item[0][0],item[0][1]), 0)
                          - unigram_p.setdefault((item[0][0],),all_sentences) for item in trigram_count}
    q_values = trigram_p
    return q_values

def calc_known(train_words):
    known_words = set([])
    words = []
    [words.extend(sentence) for sentence in train_words]
    counted = count(words)
    for word in counted:
        if word[1] > RARE_WORD_MAX_FREQ and word[0] not in known_words:
            known_words.add(word[0])
    return known_words

def replace_rare_tokens(tokens, rare_thresh=3, known_words=None):
    if known_words is None:
        known_words = dict()
        for i in range(len(tokens)):
            for j in range(len(tokens[i])):
                word = tokens[i][j][0].lower()
                known_words[word] = known_words.setdefault(word, 0) + 1
    for i in range(len(tokens)):
        for j in range(len(tokens[i])):
            word = tokens[i][j][0].lower()
            print tokens[i][j]
            if known_words.setdefault(word, 0) < rare_thresh:
                tokens[i][j][0] = RARE_SYMBOL
    return known_words

def replace_rare(train_words, known_words):
    brown_words_rare = []
    for sentence in train_words:
        modified = []
        for word in sentence:
            if word not in known_words:
                modified.append(RARE_SYMBOL)
            else:
                modified.append(word)
        brown_words_rare.append(modified)
    return brown_words_rare

def calc_emission(train_words_rare, train_tags):
    taglist = set([])
    tag_unigrams = []
    pair_tuples = []

    for i in range(0, len(train_words_rare)):
        sentence = train_words_rare[i]
        tags = train_tags[i]
        for j in range(1, len(sentence)):
            word = sentence[j]
            tag = tags[j]
            pair = (word, tag)
            if tag not in taglist:
                taglist.add(tag)
            pair_tuples.append(pair)
            tag_unigrams.append((tag,))

    counted_tag_unigrams = dict(count(tag_unigrams))
    counted_pairs = count(pair_tuples)
    all_sentences = float(len(train_words_rare))

    e_values = {pair[0] : math.log(
        float(pair[1]) / float(counted_tag_unigrams.setdefault((pair[0][1],), all_sentences)), 2)
              for pair in counted_pairs}

    return e_values, taglist

# This function takes the output from calc_emissions() and outputs it
def output_emissions(e_values, filename):
    outfile = open(filename, "w")
    emissions = e_values.keys()
    emissions.sort()
    for item in emissions:
        output = " ".join([item[0], item[1], str(e_values[item])])
        outfile.write(output + '\n')
    outfile.close()

def calc_viterbi(sentence, last_tag_probs, e_values, q_values, taglist, known_words, tagged):
    revised_tag_probs = [(START_SYMBOL, 0), (START_SYMBOL, 0)]
    for i in range(2, len(sentence)-1):
        word = sentence[i]
        last_prob = last_tag_probs[i-1][1]
        max_tag = "NN"
        max_value = 2*LOG_PROB_OF_ZERO+last_prob
        for tag in taglist:
            if word in known_words:
                word_temp = word
            else:
                word_temp = RARE_SYMBOL
            emission = e_values.setdefault((word_temp, tag), LOG_PROB_OF_ZERO)
            transmission = q_values.setdefault((revised_tag_probs[i-2][0], revised_tag_probs[i-1][0], tag), LOG_PROB_OF_ZERO)
            next1 = q_values.setdefault((revised_tag_probs[i-1][0], tag, last_tag_probs[i+1][0]), LOG_PROB_OF_ZERO)
            next2 = q_values.setdefault((tag, last_tag_probs[i+1][0], last_tag_probs[i+2][0]), LOG_PROB_OF_ZERO)
            prob = last_prob+emission+transmission+next1+next2
            if prob > max_value:
                max_value = prob
                max_tag = tag
        revised_tag_probs.append((max_tag, max_value))
        tagged.append(word+'\\'+max_tag)
        tagged.append(' ')
    tagged.append('\r\n')
    revised_tag_probs.append((STOP_SYMBOL, 0))
    revised_tag_probs.append((STOP_SYMBOL, 0))
    return revised_tag_probs

def viterbi(train_dev_words, taglist, known_words, q_values, e_values):
    tagged = []

    for sentence in train_dev_words:
        sentence.insert(0, START_SYMBOL)
        sentence.insert(0, START_SYMBOL)
        sentence.append(STOP_SYMBOL)
        tag_probs = [(START_SYMBOL, 0), (START_SYMBOL, 0)]
        for i in range(2, len(sentence)):
            word = sentence[i]
            last_prob = tag_probs[i-1][1]
            max_tag = "NN"
            max_value = 2*LOG_PROB_OF_ZERO+last_prob
            for tag in taglist:
                if word in known_words:
                    word_temp = word
                else:
                    word_temp = RARE_SYMBOL
                emission = e_values.setdefault((word_temp, tag), LOG_PROB_OF_ZERO)
                transmission = q_values.setdefault((tag_probs[i-2][0], tag_probs[i-1][0], tag), LOG_PROB_OF_ZERO)
                prob = last_prob+emission+transmission
                if prob > max_value:
                    max_value = prob
                    max_tag = tag
            tag_probs.append((max_tag, max_value))

        tag_probs.append((STOP_SYMBOL, 0))
        tag_probs.append((STOP_SYMBOL, 0))
        tagged_words = []
        calc_viterbi(sentence, tag_probs, e_values, q_values, taglist, known_words, tagged_words)

        str = ''
        for word in tagged_words:
            str += word
        tagged.append(str)

    return tagged

def nltk_tagger(training_words, training_tags, training_dev_words):
    training = [ zip(training_words[i],training_tags[i]) for i in xrange(len(training_words)) ]

    default_tagger = nltk.DefaultTagger('NN')
    unigram_tagger = nltk.UnigramTagger(training, backoff=default_tagger)
    bigram_tagger = nltk.BigramTagger(training, backoff=unigram_tagger)
    trigram_tagger = nltk.TrigramTagger(training, backoff=bigram_tagger)

    tagged = []
    for sentence in training_dev_words:
        tags = trigram_tagger.tag(sentence[2:-1])
        to_append = ""
        for tag in tags:
            to_append += tag[0]+'\\'+tag[1]+" "
        to_append += "\r\n"
        tagged.append(to_append)

    return tagged

def get_word_list(data):
    word_set = set()
    tag_set = set()
    for line in data:
        for sent in data:
            pairs = sent.strip().split()
            for pair in pairs:
                pair = pair.split('\\')
                word_set.add(pair[0].lower())
                tag_set.add(pair[1])
    return list(word_set), list(tag_set)


def get_feature_for_one(tokens, i, window, c_window=5):
    x = []
    word = tokens[i][0]
    x.append(word+'_*word*')
    x.append(word.lower()+'_*word_lower*')

    for j in range(-window, 0):
        x.append(tokens[i+j][1]+'t='+str(j))

    for j in range(-window, window+1):
        x.append(tokens[i+j][0].lower()+'w='+str(j)+'_lower')

    for j in range(-window, window+1):
        if j != 0:
            x.append(tokens[i+j][0]+'w='+str(j))

    for j in range(-window, window):
        x.append(tokens[i+j][0].lower()+'_'+
                 tokens[i+j+1][0].lower()+
                 'bi='+str(j))

    last_back = ''
    last_front = ''
    for j in range(0, c_window):
        k = len(word)-1-j
        if k > 0:
            last_back += word[k].lower()
        x.append(last_back+'_c=-'+str(j+1))
        if j < len(word):
            last_front += word[j].lower()
        x.append(last_front+'_c_bi='+str(j))
    punc = '.,?!;:\'\"()-'
    special = '@#$%^&*_+=~></\\{}[]|'
    inPunc = 'notInPunc'
    inSpecial = 'notInSpecial'
    inNum = 'notInNum'
    hasCap = 'notHasCap'
    for c in word:
        if c in punc:
            inPunc = 'inPunc'
        if c in special:
            inSpecial = 'inSpecial'
        if c.isdigit():
            inNum = 'inNum'
        if c.isupper():
            hasCap = 'hasCap'
    x.append(inPunc)
    x.append(inSpecial)
    x.append(inNum)
    x.append(hasCap)

    return x

def get_features(corpus, window, line=-1, index=-1):
    X = []
    y = []
    k = 1
    if line == -1:
        for tokens in corpus:
            #print(str(k)+' '+str(tokens))
            #print k
            if index == -1:
                for i in range(window, len(tokens)-window):
                    x = get_feature_for_one(tokens, i, window)
                    X.append(x)
                    y.append(tokens[i][1])
            else:
                x = get_feature_for_one(tokens, index, window)
                X.append(x)
                y.append(tokens[index][1])
            k += 1
    return X, y

def vectorize2(X, y):
    features = dict()
    results = dict()
    newX = []
    newY = []
    k = 1
    for x in X:
        #print k
        newx = []
        for feature in x:
            if feature not in features:
                features[feature] = len(features)+1
            newx.append(features[feature])
        newX.append(newx)
        k += 1

    for result in y:
        if result not in results:
            results[result] = len(results)
        newY.append(results[result])

    return newX, newY, features, results

def get_feature_dict(X, y):
    features = []
    for i in range(len(X[0])):
        features.append(dict())
    results = dict()
    for x in X:
        #print k
        i = 0
        for i in range(len(x)):
            feature = x[i]
            if feature not in features[i]:
                features[i][feature] = len(features[i])+1

    for result in y:
        if result not in results:
            results[result] = len(results)
    return features, results

def vectorize_one(x, y, features, results, num_features=0):
    if num_features > 0:
        k = num_features
    else:
        k = 1
        for i in range(len(features)):
            k += len(features[i])

    new_y = 0
    new_x = [0]*k
    j = 0
    for i in range(len(x)):
        if x[i] in features[i]:
            index = features[i][x[i]]
        else:
            index = 0
        new_x[j+index] = 1
        j += len(features[i])

    if y is not None and y in results:
        new_y = results[y]

    return new_x, new_y

def vectorize(X, y, features, results):
    newX = []
    newY = []
    k = 1
    for i in range(len(features)):
        k += len(features[i])
    for i in range(len(X)):
        if y is not None:
            new_x, new_y = vectorize_one(X[i], y[i], features, results, num_features=k)
        else:
            new_x, new_y = vectorize_one(X[i], None, features, results, num_features=k)
        newX.append(new_x)
        newY.append(new_y)
    return newX, newY

def train_stochastic(X, y, features, results, num_corpora=1, corpora_type_list=None, clf=None):
    if clf is None:
        clf = SGDClassifier()

    y_classes = []
    for res in results:
        y_classes.append(results[res])
    classes = np.unique(y_classes)

    k = 1
    for i in range(len(X[0])):
        k += len(features[i])

    for i in range(len(X)):
        import sys
        sys.stdout.write('%.3f%% Complete\r' % ((float(i)/float(len(X)))*100))
        x_vec, y_vec = vectorize_one(X[i], y[i], features, results, num_features=k)
        if corpora_type_list is None or num_corpora == 1:
            clf.partial_fit([x_vec], [y_vec], classes)
        else:
            x_corpora = []
            x_corpora.extend(x_vec)
            blank_results = [0]*len(x_corpora)
            for j in range(num_corpora):
                if j == corpora_type_list[i]:
                    x_corpora.extend(x_vec)
                else:
                    x_corpora.extend(blank_results)
            clf.partial_fit([x_corpora], [y_vec], classes)
    return clf


def train_tagger(X, y, type='sgdo'):
    if type == 'sgd':
        clf = SGDClassifier()
        clf.fit(X, y)
    elif type == 'sgdo':
        clf = SGDClassifier()
        classes = np.unique(y)
        for i in range(len(X)):
            sys.stdout.write('%.3f%% Complete\r' % ((float(i)/float(len(X)))*100))
            A = X[i]
            b = y[i]
            clf.partial_fit([A], [b], classes)
    elif type == 'nn':
        clf = Perceptron()
        clf.fit(X, y)
    elif type == 'nno':
        clf = Perceptron()
        classes = np.unique(y)
        for i in range(len(X)):
            sys.stdout.write('%.3f%% Complete\r' % ((float(i)/float(len(X)))*100))
            A = X[i]
            b = y[i]
            clf.partial_fit([A], [b], classes)
    elif type == 'svm':
        clf = svm.LinearSVC()
        clf.fit(X, y)
    else:
        clf = svm.LinearSVC()
        clf.fit(X, y)

    return clf

def predict_tag(X, clf):
    tag = clf.predict(X)
    return tag

def tag_sents(clf, test_set, features, results, window, num_features=0, hasher=None, selector=None):
    if hasher is None:
        from sklearn.feature_extraction import FeatureHasher
        hasher = FeatureHasher(input_type='string')
    rev_results = [None]*len(results)
    for y in results:
        rev_results[results[y]] = y

    out_sents = []
    k = 0
    for tokens in test_set:
        sent = ''
        for i in range(window, len(tokens)-window):
            word = tokens[i][0]
            test_features = get_feature_for_one(tokens, i, window)
            new_row = []
            for element in test_features:
                new_row.append(element+'_*tweet*')
            for element in test_features:
                new_row.append(element+'_*cmu*')
            vec_features = hasher.transform([new_row])
            vec_extend = vec_features

            if selector is not None:
                vec_extend, selector = select_features(vec_extend)

            pred_res = predict_tag(vec_extend, clf)[0]
            tag = rev_results[pred_res]
            tokens[i] = [word, tag]
            sent += word+'\\'+tag+' '
        out_sents.append(sent+'\r\n')
        sys.stdout.write('%.3f%% Complete\r' % ((float(k)/float(len(test_set)))*100))
        k += 1

    return out_sents

def preprocess(train_sents, window):
    sent_tokens = [None]*len(train_sents)
    for i in range(len(sent_tokens)):
        tokens = train_sents[i].split()
        for j in range(window):
            tokens.insert(0, '*\\*')
            tokens.append('STOP\\STOP')
        for j in range(len(tokens)):
            tokens[j] = list(tokens[j].split('\\'))
        sent_tokens[i] = tokens
    return sent_tokens

def select_features(X, y=None, selector=None):
    if selector is None:
        selector = SelectKBest(chi2)
    if y is None:
        x_new = selector.transform(X)
    else:
        selector = selector.fit(X, y)
        x_new = selector.transform(X)
    return X, selector

# This function takes the output of viterbi() and outputs it to file
def output_tagged(tagged, filename):
    outfile = open(filename, 'w')
    for sentence in tagged:
        outfile.write(sentence.encode('utf-8'))
    outfile.close()

def main():
    # start timer
    time.clock()

    # open training data
    infile = open("../data/gold/simple_gold_revised.txt", "r")
    train_sents = infile.readlines()
    infile.close()
    train_sents = train_sents[100:]
    # open CMU training data
    infile = open("../data/cmu_all_gold.txt")
    cmu_train_sents = infile.readlines()
    infile.close()

    window = 1

    num_corpora = 0

    sent_tokens = preprocess(train_sents, window)
    cmu_sent_tokens = preprocess(cmu_train_sents, window)
    all_tokens = sent_tokens
    all_tokens.extend(cmu_sent_tokens[:len(cmu_sent_tokens)/2])

    del train_sents
    X, y = get_features(all_tokens, window)
    print('Got Features')
    del all_tokens
    features, results = get_feature_dict(X, y)
    print('Got Feature Dict')
    X_tweets = X[0:len(sent_tokens)]
    X_cmu = X[len(sent_tokens):]
    print('Split training Data')
    print('Training on Tweets...')

    from sklearn.feature_extraction import FeatureHasher
    hasher = FeatureHasher(input_type='string')

    X_new = []
    for row in X_tweets:
        new_row = []
        new_row.extend(row)
        for element in row:
            new_row.append(element+'_*tweet*')
        X_new.append(new_row)
    for row in X_cmu:
        new_row = []
        new_row.extend(row)
        for element in row:
            new_row.append(element+'_*cmu*')
        X_new.append(new_row)

    x_vec = hasher.transform(X_new)
    y_vec = []
    for y_i in y:
        new_y = 0
        if y_i is not None and y_i in results:
            new_y = results[y_i]
        y_vec.append(new_y)

    clf = svm.LinearSVC(C=0.15)
    clf.fit(x_vec, y_vec)

    print('Done')
    print('Training on CMU...')
    print('Done')
    del X
    del y
    del sent_tokens


    ## this writes the classifier to a binary
    #from sklearn.externals import joblib
    #joblib.dump(clf, 'classifiers/cmu+gang_nn_hot.pkl')

    ## This reads the classifier from a binary
    #from sklearn.externals import joblib
    #clf = joblib.load('classifiers/cmu+gang_nn_daume.pkl')

    print('Trained Classifier')

    # open Corpus development data
#    infile = open("../data/content/content_revised_tokenized.txt", "r")
    infile = '../../data/gakirah/gakirah_aggress_loss.csv'
    print('Reading Dev')
    f = open(infile, 'rU')
    reader = csv.DictReader(f)
    train_Dev = []
    for row in reader:
        tweet = row['CONTENT'].decode('utf-8')
        train_Dev.append(tweet)

    f.close()

    train_dev_words = []
    for sentence in train_Dev:
        train_dev_words.append(sentence.rstrip().split())
    dev_tokens = [None]*len(train_Dev)
    for i in range(len(dev_tokens)):
        tokens = train_Dev[i].split()
        for j in range(window):
            tokens.insert(0, '*\\*')
            tokens.append('STOP\\STOP')
        for j in range(len(tokens)):
            tokens[j] = list(tokens[j].split('\\'))
        dev_tokens[i] = tokens

    print('Testing Dev')
    tagged_sents = tag_sents(clf, dev_tokens, features, results, window, num_corpora, hasher=hasher)
    print('Writing Results')
#    output_tagged(tagged_sents, '../results/svm_trained_on_alone+cmu.txt')
    output_tagged(tagged_sents, '../../results/pos_tagged_gakirah_aggress_loss.txt')
    print("Time: " + str(time.clock()) + ' sec')

if __name__ == "__main__": main()
import eval_parsing

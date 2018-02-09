__author__ = 'robertk'

import sys

#from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn import svm
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.feature_extraction import FeatureHasher
from sklearn.feature_extraction import DictVectorizer
import numpy as np
import math
import nltk
import word2vecReader

from scipy.sparse import csr_matrix, hstack

import gensim

START_SYMBOL = '</s>'
STOP_SYMBOL = '</s>'
RARE_SYMBOL = '_RARE_'
RARE_WORD_MAX_FREQ = 5
LOG_PROB_OF_ZERO = -1000
W2V_LEN = 400

class tagger:
    START_SYMBOL = '</s>'
    STOP_SYMBOL = '</s>'
    def __init__(self, clf_path=None, clf=None, brown_cluster_path=None, brown_cluster=None, word2vec_path=None,
                 word2vec=None, wiktionary_path=None, window=5, c_window=4):
        self.wiktionary=None
        self.clf=None
        self.tags = ['!', '#', '$', '&', '~', ',', '@', '^', 'A', 'D', 'E', 'L',
                     'M', 'N', 'O', 'P', 'R', 'S', 'T', 'U', 'V', 'X', 'Y', 'Z']
        self.tag_dict = dict()
        for i in range(len(self.tags)):
            self.tag_dict[self.tags[i]] = i

        self.brown_cluster=brown_cluster
        self.word2vec=word2vec

        if clf is not None:
            self.clf=None

        if wiktionary_path is not None:
            self.read_wiktionary(wiktionary_path)
            #print('Loaded Wictionary')

        if brown_cluster_path is not None:
            self.brown_cluster = self.read_brown_clusters(brown_cluster_path)
            #print('Loaded Brown Clusters')

        if word2vec_path is not None:
            self.word2vec = self.read_word2vec(word2vec_path)
            #self.word2vec = gensim.models.Word2Vec.load_word2vec_format(word2vec_path, binary=True)
            print('Loaded word2vec')

        self.hasher = FeatureHasher(input_type='string')
        self.window = window
        self.c_window = c_window
        self.domains = set()
        #print('Initialized Tagger')

    def read_word2vec(self, word2vec_path):
        # load the model
        file = open(word2vec_path)
        #model = gensim.models.Word2Vec.load_word2vec_format(word2vec_path, binary=False)
        model = word2vecReader.Word2Vec.load_word2vec_format(word2vec_path, binary=True)
        # Example of averaging a document
        # .. read doc
        # tokenize sents
        #vec = np.zeros(model.vector_size)
        #l = 0
        #for sent in sents:
        #    tokens = nltk.word_tokenize(sent)
        #    for token in tokens:
        #        vec += model[token]
        #        l += 1
        #vec = vec/float(l)
        #self.word2vec = model
        return model

        # now vec represents the document using the pre-trained vectors

    def read_brown_clusters(self, brown_cluster_path=None):
        if brown_cluster_path is not None:
            brown_cluster = dict()
            brown_file = open(brown_cluster_path)
            line = brown_file.readline()
            while line:
                entry = line.split()
                brown_cluster[entry[1]] = (entry[0], entry[2])
                line = brown_file.readline()
            self.brown_cluster = brown_cluster
        return self.brown_cluster

    def read_wiktionary(self, wiktionary_path=None):
        wiktionary=dict()
        alpha = '0123456789abcdefghijklmnopqrstuvwxyz'
        for c in alpha:
            #print(c)
            letter_dict = open(wiktionary_path+'/'+c)
            line = letter_dict.readline()
            while line:
                entry = line.strip().split('\t')
                if entry[0] not in wiktionary and '{{' in entry[2]:
                    import re
                    cats = re.search('\{\{(.*?)\}\}', entry[2])
                    if cats is not None:
                        cats = cats.group(1).split('|')
                        if (cats[0] == 'context' and ('Internet Slang' in cats or 'slang' in cats)) or cats[0] == 'eye dialect of':
                            wiktionary[entry[0]] = entry[1:]
                line = letter_dict.readline()

        #print('loaded')
        self.wiktionary = wiktionary
        return wiktionary


    def preprocess(self, train_sents, domains=None):
        if domains is not None:
            d = []
        else:
            d = None

        sent_tokens = [None]*len(train_sents)
        for i in range(len(sent_tokens)):
            tokens = train_sents[i].split()
            for j in range(self.window):
                tokens.insert(0, START_SYMBOL+'\\'+START_SYMBOL)
                tokens.append(STOP_SYMBOL+'\\'+STOP_SYMBOL)
            for j in range(len(tokens)):
                tokens[j] = list(tokens[j].split('\\'))
            if d is not None:
                d.extend([domains[i]]*(len(tokens)-self.window*2))
                if domains[i] not in self.domains:
                    self.domains.add(domains[i])

            sent_tokens[i] = tokens
        return sent_tokens, d

    def get_feature_for_one(self, tokens, i, domain=None):
        x = []

        word = tokens[i][0]
        x.append(word+'_word')

        # Features involved in the window

        for j in range(-5, 6):
            x.append(tokens[i+j][0].lower()+'w='+str(j)+'_lower')

        for j in range(-2, 0):
            x.append(tokens[i+j][1]+'_t='+str(j))

        for j in range(-2, 3):
            x.append(tokens[i+j][0].lower()+'_'+
                     tokens[i+j+1][0].lower()+
                     '_lower_bi='+str(j))

        # Additional Word Information
        # Brown Clusters

        for j in range(-1, 2):
            w = tokens[i+j][0]
            if w.lower() in self.brown_cluster and i+j > self.window and i+j < len(tokens)-self.window:
                brown = self.brown_cluster[w.lower()]
            else:
                brown = ('0000', '0')
            x.append(brown[0][:2]+'_brown_bin_2_='+str(j))
            x.append(brown[0][:4]+'_brown_bin_4_='+str(j))
            x.append(brown[0][:8]+'_brown_bin_8='+str(j))
            x.append(brown[0][:16]+'_brown_bin_16='+str(j))
            x.append(brown[0]+'_brown_bin='+str(j))
            x.append(brown[1]+'_brown_num')
        # word2vec
        #
        #vector = np.zeros(300)
        #if word in self.word2vec.vocab:
        #    vector = self.word2vec.syn0norm[self.word2vec.vocab[word].index]

        #x.extend(vector)

        # Character Features

        last_back = ''
        last_front = ''
        for j in range(0, self.c_window):
            k = len(word)-1-j
            if k > 0:
                last_back += word[k].lower()
            x.append(last_back+'_c=-'+str(j+1))
            if j < len(word):
                last_front += word[j].lower()
            x.append(last_front+'_c_bi='+str(j))


        # Features based on types of characters
        punc = '.,?!;:\'\"()-'
        special = '@#$%^&*_+=~></\\{}[]|'
        allPunc = 'allPunc'
        inSpecial = 'notInSpecial'
        inNum = 'notInNum'
        isCap = 'notIsCap'
        hasCap = 'notHasCap'
        if len(word) > 0:
            c = word[0]
            if c.isupper():
                isCap = 'isCap'
        hasLetter = 'notHasLetter'
        for c in word:
            if c in punc:
                allPunc = 'notAllPunc'
            if c in special:
                inSpecial = 'inSpecial'
            if c.isdigit():
                inNum = 'inNum'
            if c.isupper():
                hasCap = 'hasCap'
            if c.isalpha():
                hasLetter = 'hasLetter'
        x.append(inSpecial)
        x.append(hasCap)
        x.append(allPunc)
        x.append(hasLetter)

        #x.append(inNum)
        #x.append(isCap)


        #x.append(allPunc)
        #x.append('len='+str(len(word)))

        if self.wiktionary is not None and word.lower() in self.wiktionary:
            wiki_tag = self.wiktionary[word.lower()][0]
        else:
            wiki_tag = 'EMPTY'
        '''
        x_new = []

        if domain is not None:
            for feat in x:
                x_new.append(feat+'_'+str(domain))
            x.extend(x_new)
        x_hashed = self.hasher.transform([x])
        x_vec = x_hashed.toarray()
        '''


        #x.append(wiki_tag+'_wiki_tag')

        return x

    #def get_feature_dict(self, X, y):
    #    results = dict()
    #    for result in y:
    #        if result not in results:
    #            results[result] = len(results)
    #    return results

    def get_features(self, corpus, domains=None, line=-1, index=-1):
        X = []
        y = []
        d = None
        if line == -1:
            for k in range(len(corpus)):
                tokens = corpus[k]
                if domains is not None and len(domains) == len(corpus):
                    d = domains[k]
                if index == -1:
                    for i in range(self.window, len(tokens)-self.window):
                        x = self.get_feature_for_one(tokens, i, d)
                        X.append(x)
                        y.append(tokens[i][1])
                else:
                    x = self.get_feature_for_one(tokens, index, d)
                    X.append(x)
                    y.append(tokens[index][1])
        return X, y

    def train(self, sents, domain_list=None, window=-1):
        #self.clf = svm.SVC(C=0.2, degree=2)
        self.clf = svm.LinearSVC(C=0.2)
        if window > 0:
            self.window = window
        sents_proc, d = self.preprocess(sents, domains=domain_list)
        X, y = self.get_features(sents_proc, domains=domain_list)
        #features, self.tag_dict = self.get_feature_dict(X, y)
        #self.tag_dict = self.get_feature_dict(X, y)

        if d is not None and len(d) == len(X):
            for i in range(len(X)):
                #for j in range(len(X[i])):
                #    X[i].append(X[i][j]+'_'+str(d[i]))
                X[i].append(str(d[i]))
        else:
            self.domains = None


        #X_hash = self.hasher.transform(X)
        #X_vec = X_hash.todense()
        X_vec = self.hasher.transform(X)
        y_vec = []
        for y_i in y:
            new_y = 0
            if y_i is not None and y_i in self.tag_dict:
                new_y = self.tag_dict[y_i]
            y_vec.append(new_y)

        if self.word2vec is not None:
            word2vecCol = np.zeros((X_vec.shape[0], W2V_LEN))
            i = 0
            for X_i in X:
                word = X_i[0][:-5].lower()
                vector = np.zeros(W2V_LEN)
                if word in self.word2vec.vocab:
                    vector = self.word2vec.syn0norm[self.word2vec.vocab[word].index]
                for j in range(W2V_LEN):
                    word2vecCol[i][j] = vector[j]
                i += 1
            word2vecCol = csr_matrix(word2vecCol)
                #X.extend(vector)
            X_vec = hstack((X_vec, word2vecCol))

        print('Preprocessing Done')
        self.clf.fit(X_vec, y_vec)
        print('Trained Classifier')
        return self.clf


    def predict_tag(self, X):
        tag = self.clf.predict(X)
        return tag

    def tag_tweet(self, tokens, rev_results, hasher=None):
        tagged = []
        if hasher is None:
            from sklearn.feature_extraction import FeatureHasher
            from sklearn.feature_extraction.text import CountVectorizer
            hasher = FeatureHasher(input_type='string')
            #hasher = CountVectorizer()

        for word in tokens:
            tagged.append([word])

        tagged.insert(0, [START_SYMBOL,START_SYMBOL])
        tagged.append([STOP_SYMBOL])
        for i in range(self.window, len(tagged)-self.window):
            word = tagged[i][0]
            test_features = self.get_feature_for_one(tagged, i)
            new_row = []
            for element in test_features:
                new_row.append(element+'_*tweet*')
            for element in test_features:
                new_row.append(element+'_*cmu*')
            vec_features = hasher.transform([new_row])
            vec_extend = vec_features

            #if selector is not None:
            #    vec_extend, selector = select_features(vec_extend)
            pred_res = self.predict_tag(vec_extend)[0]
            tag = rev_results[pred_res]
            tagged[i] = [word, tag]
        return tagged[1:-1]

    def tag_sents(self, test_set, sent_domain):
        #rev_results = [None]*len(self.tag_dict)
        #for y in self.tag_dict:
        #    rev_results[self.tag_dict[y]] = y

        rev_results = self.tags

        out_sents = []
        k = 0
        for tokens in test_set:
            sent = ''

            for i in range(self.window, len(tokens)-self.window):
                word = tokens[i][0]
                test_features = self.get_feature_for_one(tokens, i)
                new_row = []
                #if self.domains is not None:
                    #for domain in self.domains:
                #    for element in test_features:
                #        #new_row.append(element+'_'+domain)
                #        new_row.append(element+'_'+'*cmu*')
                #new_row.extend(test_features)
                #    vec_features = self.hasher.transform([new_row])
                #else:
                #    vec_features = self.hasher.transform([test_features])
                test_features.append(sent_domain)
                vec_features = self.hasher.transform([test_features])

                if self.word2vec is not None:
                    word = test_features[0][:-5].lower()
                    vector = np.zeros(W2V_LEN)
                    if word in self.word2vec.vocab:
                        vector = self.word2vec.syn0norm[self.word2vec.vocab[word].index]
                    vector = csr_matrix(vector)


                    vec_features = hstack((vec_features, vector))


                pred_res = self.predict_tag(vec_features)[0]
                tag = rev_results[pred_res]
                tokens[i] = [word, tag]
                sent += word+'\\'+tag+' '
            out_sents.append(sent+'\r\n')
            #sys.stdout.write('%.3f%% Complete\r' % ((float(k)/float(len(test_set)))*100))
            k += 1

        return out_sents

    def convert_conll(self, sents):
        conll = []
        for sent in sents:
            tokens = sent.split()
            i = 1
            for token in tokens:
                token = token.split('\\')
                line = str(i) + '\t'+token[0]+'\t_\t'+token[1]+'\t'+token[1]+'\t_\t0\t_\t_\t_'
                conll.append(line)
                i += 1
            conll.append('')
        return conll

    def output_tagged(self, tagged, filename=None):
        if filename is not None:
            outfile = open(filename, 'w')
            for sentence in tagged:
                outfile.write(sentence)
            outfile.close()
        else:
            for sentence in tagged:
                print(sentence)

    def cross_validation(self, sents, domains, end_train=-1, folds=4):
        tagged_sents = []
        if end_train < 0:
            end_train = len(sents)
        size_dev = int(end_train/folds)

        index = 0
        for i in range(folds):
            next = index + size_dev
            if i == folds-1:
                next = end_train
            train_sents = []
            train_sents.extend(sents[:index])
            train_sents.extend(sents[next:])
            train_domains = []
            train_domains.extend(domains[:index])
            train_domains.extend(domains[next:])

            self.train(train_sents, train_domains)

            dev_sents = sents[index:next]
            dev_tokens, _ = self.preprocess(dev_sents)
            tagged_sents.extend(self.tag_sents(dev_tokens, domains[i]))
            index = next

        return tagged_sents

    def save_clf(self, path='../classifiers/POS-tagger.pkl'):
        # this writes the classifier to a binary
        from sklearn.externals import joblib
        joblib.dump(self.clf, path)

    def load_clf(self, path='../classifiers/POS-tagger.pkl'):
        # This reads the classifier from a binary
        from sklearn.externals import joblib
        self.clf = joblib.load(path)
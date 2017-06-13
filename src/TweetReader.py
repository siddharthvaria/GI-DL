import codecs
import sys
import math
import numpy as np
from utils import unicode_csv_reader2
from collections import defaultdict
from sklearn.model_selection import StratifiedShuffleSplit

class TweetCorpus:
    '''Simple corpus reader.'''
    '''
    Format of tweets_file (comma separated) is:
    Columns on first line followed by actual tweets from next line
    Columns: AUTHOR,CONTENT,LABEL,DATE,URL,DESC
    '''
    '''Each tweet is stored as a dictionary'''
    '''
    The collapsed aggression code contained examples of insults, threats, bragging, hypervigilance
    and challenges with authority
    '''
    '''
    The collapsed grief code included examples of distress, sadness, loneliness
    and death.
    '''
    def __init__(self, train_file, val_file, test_file):

        self.max_len = 0
        self.aggress = set(['aggress', 'insult', 'snitch', 'threat', 'brag', 'aod', 'aware', 'authority', 'trust', 'fight', 'pride', 'power', 'lyric'])
        self.loss = set(['loss', 'grief', 'death', 'sad', 'alone', 'reac', 'guns'])
        # self.other = set(['deleted/ sex', 'money', 'gen, rel', 'rel, wom', 'authenticity', 'anger', 'retweet', 'wom', 'convo/neighborhood', 'gen, money', 'gen/women', 'deleted', 'gen/location', 'rel', 'indentity', 'amiable?', 'happy', 'sex', 'promo', 'mention', 'gen, happy', 'general', 'gen', 'identity', 'rel/gen', 'convo', 'joke', 'trans', 'wom, rel'])

        self.label2idx = {'aggress':0, 'loss':1, 'other':2}
        self.idx2label = {id: c for c, id in self.label2idx.iteritems()}

        self.train_tweets = self.read_tweets(train_file)
        self.val_tweets = self.read_tweets(val_file)
        self.test_tweets = self.read_tweets(test_file)

        self.char2idx = None
        self.idx2char = None
        self.init_char_dictionaries()

        self.train_label_dist = None
        self.val_label_dist = None
        self.test_label_dist = None
        self.init_label_dists()


#     def read_tweets(self, file_name):
#
#         if file_name is None:
#             return None
#
#         tweets = []
#         with codecs.open(file_name, "r", "utf-8") as fh:
#             columns = fh.readline().strip().split(',')
#             # print columns
#             for line in fh:
#                 entries = line.strip().split(',')
#                 if len(entries) != len(columns):
#                     continue
#                 # print len(tweets), len(entries)
#                 tweet = {}
#                 for idx, column in enumerate(columns):
#                     tweet[column] = entries[idx]
#                 tweets.append(tweet)
#
#         return tweets

    def collapsed_label(self, fine_grained):
        fine_grained = fine_grained.lower()
        for a in self.aggress:
            if a in fine_grained: return 'aggress'
        for l in self.loss:
            if l in fine_grained: return 'loss'
        else: return 'other'

    def read_tweets(self, file_name):

        if file_name is None:
            return None

        tweets = []

        with open(file_name) as fh:

            reader = unicode_csv_reader2(fh)

            for row in reader:

                if row['CONTENT'] in (None, ''): continue

                label = ''
                if 'LABEL' in row.keys():
                    label = self.collapsed_label(row['LABEL'].lower())
                    row['LABEL'] = label

                tweets.append(row)

        return tweets

    def write_tweets(self, file_name, tweets, columns):

        with codecs.open(file_name, "w", "utf-8") as fh:
            fh.write(','.join(columns))
            fh.write('\n')
            for tweet in tweets:
                _tmp = []
                for column in columns:
                    if column in tweet:
                        _tmp.append(tweet[column])
                    else:
                        _tmp.append('')
                fh.write(','.join(_tmp))
                fh.write('\n')

    def init_char_dictionaries(self):

        if self.train_tweets is None or self.val_tweets is None or self.test_tweets is None:
            return

        self.char2idx = defaultdict(int)

        # self.char2idx[''] = 0

        self._update_char2idx(self.train_tweets)
        self._update_char2idx(self.val_tweets)
        self._update_char2idx(self.test_tweets)

        self.idx2char = {id: c for c, id in self.char2idx.iteritems()}

    def _update_char2idx(self, tweets):

        for tweet in tweets:
            if 'CONTENT' in tweet:
                content = tweet['CONTENT'].strip()
                self.max_len = max(self.max_len, len(content))
                for char in content:
                    if char not in self.char2idx:
                        self.char2idx[char] = len(self.char2idx) + 1

    def init_label_dists(self):

        if self.train_tweets is None or self.val_tweets is None or self.test_tweets is None:
            return

        self.train_label_dist = defaultdict(int)
        self.val_label_dist = defaultdict(int)
        self.test_label_dist = defaultdict(int)

        self._update_label_dist(self.train_tweets, self.train_label_dist)
        self._update_label_dist(self.val_tweets, self.val_label_dist)
        self._update_label_dist(self.test_tweets, self.test_label_dist)

    def _update_label_dist(self, tweets, label_dist):
        for tweet in tweets:
            if 'LABEL' in tweet:
                label_dist[tweet['LABEL']] += 1

    def tweet2Indices(self, tweet):
        indices = [self.char2idx[c] for c in tweet['CONTENT']]
        return np.asarray(indices + [0 for _ in xrange(self.max_len - len(indices))])

    def label2Index(self, tweet):
        return self.label2idx[tweet['LABEL']]

    def get_stratified_splits(self, split_ratio = 0.2):

        X = []
        y = []

        for tweet in self.train_tweets:
            X.append(self.tweet2Indices(tweet))
            y.append(self.label2Index(tweet))

        for tweet in self.val_tweets:
            X.append(self.tweet2Indices(tweet))
            y.append(self.label2Index(tweet))

        for tweet in self.test_tweets:
            X.append(self.tweet2Indices(tweet))
            y.append(self.label2Index(tweet))

        X = np.asarray(X)
        y = np.asarray(y)

#         print 'type(X): ', type(X)
#         print 'type(X[0]): ', type(X[0])
#         print 'type(y): ', type(y)
#         print 'type(y[0]): ', type(y[0])
#         print 'X.shape', X.shape

        X_train, X_test, y_train, y_test = self._perform_stratified_shuffle_split(X, y, split_ratio = split_ratio)
        X_train, X_val, y_train, y_val = self._perform_stratified_shuffle_split(X_train, y_train, split_ratio = math.floor((split_ratio * 100) / (1.0 - split_ratio)) / 100)

        return X_train, X_val, X_test, y_train, y_val, y_test

    def _perform_stratified_shuffle_split(self, X, y, split_ratio = 0.2):

        sss = StratifiedShuffleSplit(n_splits = 1, test_size = split_ratio, random_state = 0)

        for train_index, test_index in sss.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

        return X_train, X_test, y_train, y_test

    def get_label_dist(self, y_train, y_val, y_test):

        train_label_dist = defaultdict(int)
        val_label_dist = defaultdict(int)
        test_label_dist = defaultdict(int)

        for y in y_train:
            train_label_dist[y] += 1

        for y in y_val:
            val_label_dist[y] += 1

        for y in y_test:
            test_label_dist[y] += 1

        return train_label_dist, val_label_dist, test_label_dist

import codecs
import csv
import math
import numpy as np
from utils import unicode_csv_reader2
from collections import defaultdict
from sklearn.model_selection import StratifiedShuffleSplit
import cStringIO

class UnicodeWriter:
    """
    A CSV writer which will write rows to CSV file "f",
    which is encoded in the given encoding.
    """

    def __init__(self, f, dialect = csv.excel, encoding = "utf-8", **kwds):
        # Redirect output to a queue
        self.queue = cStringIO.StringIO()
        self.writer = csv.writer(self.queue, dialect = dialect, **kwds)
        self.stream = f
        self.encoder = codecs.getincrementalencoder(encoding)()

    def writerow(self, row):
        self.writer.writerow([s.encode("utf-8") for s in row])
        # Fetch UTF-8 output from the queue ...
        data = self.queue.getvalue()
        data = data.decode("utf-8")
        # ... and reencode it into the target encoding
        data = self.encoder.encode(data)
        # write to the target stream
        self.stream.write(data)
        # empty queue
        self.queue.truncate(0)

    def writerows(self, rows):
        for row in rows:
            self.writerow(row)

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
    def __init__(self, train_file = None, val_file = None, test_file = None, unlabeled_tweets_file = None):

        self.max_len = 0
        self.len_dict = None

        self.aggress = set(['aggress', 'insult', 'snitch', 'threat', 'brag', 'aod', 'aware', 'authority', 'trust', 'fight', 'pride', 'power', 'lyric'])
        self.loss = set(['loss', 'grief', 'death', 'sad', 'alone', 'reac', 'guns'])
        # self.other = set(['deleted/ sex', 'money', 'gen, rel', 'rel, wom', 'authenticity', 'anger', 'retweet', 'wom', 'convo/neighborhood', 'gen, money', 'gen/women', 'deleted', 'gen/location', 'rel', 'indentity', 'amiable?', 'happy', 'sex', 'promo', 'mention', 'gen, happy', 'general', 'gen', 'identity', 'rel/gen', 'convo', 'joke', 'trans', 'wom, rel'])

        self.train_tweets = self.read_tweets(train_file, 'CONTENT', ',')
        self.val_tweets = self.read_tweets(val_file, 'CONTENT', ',')
        self.test_tweets = self.read_tweets(test_file, 'CONTENT', ',')
        self.unlabeled_tweets = self.read_tweets(unlabeled_tweets_file, 'text', ',')

        self.char2idx = None
        self.idx2char = None
        self.init_char_dictionaries()

        self.label2idx = None
        self.idx2label = None
        self.init_label_dictionaries()

    def collapsed_label(self, fine_grained):
        fine_grained = fine_grained.lower()
        for a in self.aggress:
            if a in fine_grained: return 'aggress'
        for l in self.loss:
            if l in fine_grained: return 'loss'
        else: return 'other'

    def read_tweets(self, file_name, column_name, delimiter):

        if file_name is None:
            return None

        tweets = []

        line_count = 0

        with open(file_name) as fh:

            reader = unicode_csv_reader2(fh, delimiter = delimiter)

            for row in reader:

                line_count += 1

                if row[column_name] in (None, ''): continue

                # preprocess the tweet
                # row[column_name] = preprocess(row[column_name])

                # put a hard cutoff of 150 characters
                if len(row[column_name]) > 150:
                    continue

                if column_name == 'CONTENT':
                    if 'LABEL' in row.keys():
                        label = self.collapsed_label(row['LABEL'].lower())
                        row['LABEL'] = label

                tweets.append(row)

        return tweets

    def write_tweets(self, file_name, tweets, columns):

        if file_name is None or tweets is None:
            return None

        unicode_writer = UnicodeWriter(open(file_name, 'w'))
        unicode_writer.writerow(columns)
        for tweet in tweets:
            _tmp = []
            for column in columns:
                if column in tweet:
                    _tmp.append(tweet[column])
                else:
                    _tmp.append('')
            unicode_writer.writerow(_tmp)

#     def preprocess_tweet(self, text):
#
#         # get rid of continuous underlines in the tweet
#         text = re.sub(r"_", "", text)
#         # get rid of <a href=""> html tags
#         text = re.sub(r"<a.*?>", "", text)
#         # get rid of urls
#         text = re.sub(r"http\S+", "", text)
#
#         return text.strip()

    def init_char_dictionaries(self):

        self.char2idx = defaultdict(int)

        self.len_dict = defaultdict(int)

        self._update_char2idx(self.train_tweets, 'CONTENT')
        self._update_char2idx(self.val_tweets, 'CONTENT')
        self._update_char2idx(self.test_tweets, 'CONTENT')
        self._update_char2idx(self.unlabeled_tweets, 'text')

        self.idx2char = {id: c for c, id in self.char2idx.iteritems()}

    def _update_char2idx(self, tweets, column_name):

        if tweets is None:
            return

        for tweet in tweets:
            if column_name in tweet:
                content = tweet[column_name]
                self.len_dict[len(content)] += 1
                self.max_len = max(self.max_len, len(content))
                for char in content:
                    if char not in self.char2idx:
                        self.char2idx[char] = len(self.char2idx) + 1

    def init_label_dictionaries(self):

        self.label2idx = defaultdict(int)

        self._update_label2idx(self.train_tweets)
        self._update_label2idx(self.val_tweets)
        self._update_label2idx(self.test_tweets)

        self.idx2label = {id: c for c, id in self.label2idx.iteritems()}

    def _update_label2idx(self, tweets):

        if tweets is None:
            return

        for tweet in tweets:
            if 'LABEL' in tweet:
                label = tweet['LABEL']
                if label not in self.label2idx:
                    self.label2idx[label] = len(self.label2idx)

    def get_class_names(self):

        class_names = []
        for idx in xrange(len(self.idx2label)):
            class_names.append(self.idx2label[idx])

        return class_names

    def tweet2Indices(self, tweet, column_name):

        indices = [self.char2idx[c] for c in tweet[column_name]]
        return np.asarray([0 for _ in xrange(self.max_len - len(indices))] + indices)

    def label2Index(self, tweet):
        return self.label2idx[tweet['LABEL']]

    def get_splits(self):

        X_train = []
        X_val = []
        X_test = []
        y_train = []
        y_val = []
        y_test = []

        if self.train_tweets is not None:
            for tweet in self.train_tweets:
                X_train.append(self.tweet2Indices(tweet, 'CONTENT'))
                y_train.append(self.label2Index(tweet))

        if self.val_tweets is not None:
            for tweet in self.val_tweets:
                X_val.append(self.tweet2Indices(tweet, 'CONTENT'))
                y_val.append(self.label2Index(tweet))

        if self.test_tweets is not None:
            for tweet in self.test_tweets:
                X_test.append(self.tweet2Indices(tweet, 'CONTENT'))
                y_test.append(self.label2Index(tweet))

        X_train = np.asarray(X_train)
        y_train = np.asarray(y_train)
        X_val = np.asarray(X_val)
        y_val = np.asarray(y_val)
        X_test = np.asarray(X_test)
        y_test = np.asarray(y_test)

        return X_train, X_val, X_test, y_train, y_val, y_test

    def get_splits_for_lm1(self):

        X = []

        if self.unlabeled_tweets is not None:
            for tweet in self.unlabeled_tweets:
                X.append(self.tweet2Indices(tweet, 'text'))

        X = np.asarray(X)

        return X

    def get_splits_for_lm2(self):

        X = []

        if self.train_tweets is not None:
            for tweet in self.train_tweets:
                X.append(self.tweet2Indices(tweet, 'CONTENT'))

        if self.val_tweets is not None:
            for tweet in self.val_tweets:
                X.append(self.tweet2Indices(tweet, 'CONTENT'))

        if self.test_tweets is not None:
            for tweet in self.test_tweets:
                X.append(self.tweet2Indices(tweet, 'CONTENT'))

        X = np.asarray(X)

        return X

    def get_stratified_splits(self, split_ratio = 0.2):

        X = []
        y = []

        if self.train_tweets is not None:
            for tweet in self.train_tweets:
                X.append(self.tweet2Indices(tweet, 'CONTENT'))
                y.append(self.label2Index(tweet))

        if self.val_tweets is not None:
            for tweet in self.val_tweets:
                X.append(self.tweet2Indices(tweet, 'CONTENT'))
                y.append(self.label2Index(tweet))

        if self.test_tweets is not None:
            for tweet in self.test_tweets:
                X.append(self.tweet2Indices(tweet, 'CONTENT'))
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

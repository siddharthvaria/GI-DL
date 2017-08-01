from data_utils.utils import unicode_csv_reader2
from collections import defaultdict
import os
from cStringIO import StringIO
import argparse
import cPickle as pickle
import datetime
import numpy as np
import string, re
from random import randint
# from data_utils.preprocess import preprocess

aggress = set(['aggress', 'insult', 'snitch', 'threat', 'brag', 'aod', 'aware', 'authority', 'trust', 'fight', 'pride', 'power', 'lyric'])
loss = set(['loss', 'grief', 'death', 'sad', 'alone', 'reac', 'guns'])
regex = re.compile('[%s]' % re.escape(string.punctuation))

def get_delimiter(data_file):
    if data_file.endswith('.csv'):
        delimiter = ','
    elif data_file.endswith('.tsv'):
        delimiter = '\t'
    elif data_file.endswith('.txt'):
        delimiter = ' '
    return delimiter

def collapse_label(fine_grained):
    fine_grained = fine_grained.lower()
    for a in aggress:
        if a in fine_grained: return 'aggress'
    for l in loss:
        if l in fine_grained: return 'loss'

    return 'other'

def preprocess(text):

    # remove emoji placeholders
    text = re.sub('(::emoji::)|#|', '', text.lower())
    # replace user handles with a constant
    text = re.sub('@[0-9a-zA-Z_]+', 'USER_HANDLE', text)
    # replace urls
    text = re.sub('https?://[a-zA-Z0-9_\./]*', 'URL', text)
    # remove punctuations
    text = regex.sub(' ', text)
    # remove extra white space due to above operations
    text = re.sub(' +', ' ', text)
    return text

def parse_line(line, text_column, label_column, max_len, normalize = False):

    # take line (dict) as input and return text along with label if label is present
    if line[text_column] in (None, ''):
        return None, None

    if normalize:
        line[text_column] = preprocess(line[text_column])

    if len(line[text_column]) > max_len:
        line[text_column] = line[text_column][:max_len]

    X_c = line[text_column]
    if label_column in line.keys():
        y_c = collapse_label(line[label_column])
        # y_c = line[label_column]
    else:
        y_c = None

    return X_c, y_c

class TweetPreprocessor:

    def __init__(self, time_stamp, max_len):
        self.time_stamp = time_stamp
        self.max_len = max_len
        self.char2idx = defaultdict(int)
        self.label2idx = defaultdict(int)
        self.len_dict = defaultdict(int)
        self.class2count = defaultdict(int)

    def datum_to_string(self, X_ids, y_id):

        file_str = StringIO()
        file_str.write(','.join(X_ids))
        file_str.write('<:>')
        file_str.write(y_id)
        return file_str.getvalue()

    def read_data(self, output_files_dir, data_file, parser, text_column, label_column, normalize = False, is_train = False):

        if data_file is None:
            return

        _, fname = os.path.split(data_file)
        dot_index = fname.rindex('.')
        fname_wo_ext = fname[:dot_index]

        new_data_file = os.path.join(output_files_dir, fname_wo_ext + '_' + self.time_stamp + '.txt')

        delimiter = get_delimiter(data_file)
        line_count = 0
        with open(data_file, 'r') as fhr, open(new_data_file, 'w') as fhw:
            reader = unicode_csv_reader2(fhr, delimiter = delimiter)
            for row in reader:
                line_count += 1
                X_c, y_c = parser(row, text_column, label_column, self.max_len, normalize = normalize)
                if X_c is None:
                    continue
                X_ids = self.update_char2idx(X_c)
                # _tmp = ','.join(X_ids)
                if y_c is not None:
                    y_id = self.update_label2idx(y_c)
                else:
                    y_id = ''

                # update class dictionary
                if is_train and y_id != '':
                    self.class2count[int(y_id)] += 1

                fhw.write(self.datum_to_string(X_ids, y_id))
                fhw.write('\n')

    def update_char2idx(self, tweet):

        if tweet is None:
            return None

        idx = []

        self.len_dict[len(tweet)] += 1

        for char in tweet:
            if char not in self.char2idx:
                self.char2idx[char] = len(self.char2idx) + 1
            idx.append(str(self.char2idx[char]))

        return idx

    def update_label2idx(self, label):

        if label is None:
            return None

        if label not in self.label2idx:
            self.label2idx[label] = len(self.label2idx)

        return str(self.label2idx[label])

    def get_class_weights(self):

        total = 0.0
        for cls in self.class2count.keys():
            total += (float(1) / self.class2count[cls])

        K = float(1) / total

        n_classes = len(self.label2idx)
        class_weights = {}
        for i in xrange(n_classes):
            class_weights[i] = (K / self.class2count[i])

        return class_weights

    def split_unlabeled_data(self, output_files_dir, data_file, split_ratio = 0.2):

        if data_file is None:
            return

        _, fname = os.path.split(data_file)
        dot_index = fname.rindex('.')
        fname_wo_ext = fname[:dot_index]

        all_data_file = os.path.join(output_files_dir, fname_wo_ext + '_' + self.time_stamp + '.txt')
        tr_data_file = os.path.join(output_files_dir, fname_wo_ext + '_tr_' + self.time_stamp + '.txt')
        val_data_file = os.path.join(output_files_dir, fname_wo_ext + '_val_' + self.time_stamp + '.txt')

        rnd_p = np.random.permutation(10)
        tr_index = int((1 - split_ratio) * 10)
        tr_indices = rnd_p[:tr_index]

        with open(all_data_file, 'r') as fhr, open(tr_data_file, 'w') as fhw1, open(val_data_file, 'w') as fhw2:
            for line in fhr:
                rv = randint(0, 9)
                if rv in tr_indices:
                    fhw1.write(line)
                else:
                    fhw2.write(line)

    def print_stats(self):
        print 'Number of unique characters: ', len(self.char2idx) + 1
        print 'Number of classes: ', len(self.label2idx)
        print 'Length distribution: ', self.len_dict
        print 'Class distribution: ', self.class2count

def main(args):

    print 'Normalize Tweets: ', args['normalize']
    time_stamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    tweet_preprocessor = TweetPreprocessor(time_stamp, max_len = 140)
    print 'Processing training set . . .'
    tweet_preprocessor.read_data(args['output_file_dir'], args['train_file'], parse_line, 'CONTENT', 'LABEL', normalize = args['normalize'], is_train = True)
    print 'Processing validation set . . .'
    tweet_preprocessor.read_data(args['output_file_dir'], args['val_file'], parse_line, 'CONTENT', 'LABEL', normalize = args['normalize'])
    print 'Processing test set . . .'
    tweet_preprocessor.read_data(args['output_file_dir'], args['test_file'], parse_line, 'CONTENT', 'LABEL', normalize = args['normalize'])
    print 'Processing unlabeled set . . .'
    tweet_preprocessor.read_data(args['output_file_dir'], args['tweets_file'], parse_line, 'text', '', normalize = args['normalize'])
    tweet_preprocessor.print_stats()
    weights = tweet_preprocessor.get_class_weights()
    tweet_preprocessor.split_unlabeled_data(args['output_file_dir'], args['tweets_file'], split_ratio = 0.2)
    pickle.dump([tweet_preprocessor.char2idx, tweet_preprocessor.label2idx, weights, tweet_preprocessor.max_len], open(os.path.join(args['output_file_dir'], 'dictionaries_' + time_stamp + '.p'), "wb"))

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = '')
    parser.add_argument('train_file', type = str)
    parser.add_argument('val_file', type = str)
    parser.add_argument('test_file', type = str)
    parser.add_argument('--tweets_file', type = str, default = None)
    parser.add_argument('--normalize', type = bool, default = False)
    parser.add_argument('output_file_dir', type = str)

    args = vars(parser.parse_args())

    main(args)

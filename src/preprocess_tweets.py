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
import codecs
# from data_utils.preprocess import preprocess

aggress = set(['aggress', 'insult', 'snitch', 'threat', 'brag', 'aod', 'aware', 'authority', 'trust', 'fight', 'pride', 'power', 'lyric'])
loss = set(['loss', 'grief', 'death', 'sad', 'alone', 'reac', 'guns'])
regex = re.compile('[%s]' % re.escape(string.punctuation))
regex_digit = re.compile(r"[+-]?\d+(?:\.\d+)?")

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
    # replace all other white space with a single space
    text = re.sub('\s+', ' ', text)
    # remove emoji placeholders
    text = re.sub('(::emoji::)|#|', '', text.lower())
    # replace user handles with a constant
    text = re.sub('@[0-9a-zA-Z_]+', 'USER_HANDLE', text)
    # replace urls
    text = re.sub('https?://[a-zA-Z0-9_\./]*', 'URL', text)
#     # remove punctuations
#     text = regex.sub(' ', text)
#     # remove digits below
#     text = regex_digit.sub(' ', text)
    # remove extra white space due to above operations
    text = re.sub(' +', ' ', text)
    return text

def parse_line(line, text_column, label_column, max_len, stop_chars = None, normalize = False, add_ss_markers = False, word_level = False):

    if add_ss_markers:
        # -4 to account for start and stop markers and spaces
        _max_len = max_len - 4
    else:
        _max_len = max_len

    # take line (dict) as input and return text along with label if label is present
    if line[text_column] in (None, ''):
        return None, None

    if stop_chars is not None:
        line[text_column] = [ch for ch in line[text_column] if ch not in stop_chars]
        line[text_column] = ''.join(line[text_column])

    if normalize:
        line[text_column] = preprocess(line[text_column])

    # print line[text_column]

    if line[text_column] in (None, ''):
        return None, None

    if word_level:
        line[text_column] = line[text_column].split(' ')

    if len(line[text_column]) > _max_len:
        line[text_column] = line[text_column][:_max_len]

    if add_ss_markers:
        line[text_column] = '< ' + line[text_column] + ' >'

    X_c = line[text_column]
    if label_column in line.keys():
        # y_c = collapse_label(line[label_column])
        y_c = line[label_column]
    else:
        y_c = None

    return X_c, y_c

def read_stop_chars(stop_chars_file):
    if stop_chars_file is None:
        return None
    stop_chars = []
    with codecs.open(stop_chars_file, 'r', encoding = 'utf-8') as fh:
        for line in fh:
            ch = line.split('\t')[0].strip()
            stop_chars.append(ch)
    return set(stop_chars)

class TweetPreprocessor:

    def __init__(self, stop_chars, time_stamp, max_len, word_level, normalize, add_ss_markers):

        self.stop_chars = stop_chars
        self.time_stamp = time_stamp
        self.max_len = max_len
        self.word_level = word_level
        self.normalize = normalize
        self.add_ss_markers = add_ss_markers
        self.char2idx = defaultdict(int)

        # add start and stop markers
        if self.add_ss_markers:
            self.char2idx['<'] = 1
            self.char2idx['>'] = 2

        self.label2idx = defaultdict(int)
        self.len_dict = defaultdict(int)
        self.class2count = defaultdict(int)

    def datum_to_string(self, X_ids, y_id):

        file_str = StringIO()
        file_str.write(','.join(X_ids).strip())
        file_str.write('<:>')
        file_str.write(y_id)
        return file_str.getvalue()

    def read_data(self, output_files_dir, data_file, parser, text_column, label_column, is_train = False):

        if data_file is None:
            return

        _, fname = os.path.split(data_file)
        dot_index = fname.rindex('.')
        fname_wo_ext = fname[:dot_index]

        indices_file = os.path.join(output_files_dir, fname_wo_ext + '_' + self.time_stamp + '.txt')
        new_data_file = os.path.join(output_files_dir, fname_wo_ext + '_pp_.txt')

        delimiter = get_delimiter(data_file)
        line_count = 0
        with open(data_file, 'r') as fhr, open(indices_file, 'w') as fhw1, codecs.open(new_data_file, 'w', encoding = 'utf-8') as fhw2:
            reader = unicode_csv_reader2(fhr, delimiter = delimiter)
            for row in reader:
                line_count += 1
                fhw2.write('###########################################################################')
                fhw2.write('\n')
                fhw2.write(row[text_column])
                fhw2.write('\n')
                # optional parameters to the parser
                # stop_chars = None, normalize = False, add_ss_markers = False, word_level = False
                X_c, y_c = parser(row, text_column, label_column, self.max_len, stop_chars = self.stop_chars, normalize = self.normalize, word_level = self.word_level, add_ss_markers = self.add_ss_markers)
                if X_c is None:
                    fhw2.write('None')
                    fhw2.write('\n')
                    fhw2.write('###########################################################################')
                    fhw2.write('\n')
                    continue
                fhw2.write(''.join(X_c))
                fhw2.write('\n')
                fhw2.write('###########################################################################')
                fhw2.write('\n')
                X_ids = self.update_char2idx(X_c)
                # _tmp = ','.join(X_ids)
                if y_c is not None:
                    y_id = self.update_label2idx(y_c)
                else:
                    y_id = ''

                # update class dictionary
                if is_train and y_id != '':
                    self.class2count[int(y_id)] += 1

                fhw1.write(self.datum_to_string(X_ids, y_id))
                fhw1.write('\n')

    def update_char2idx(self, tweet):

        if tweet is None:
            return None

        idx = []

        self.len_dict[len(tweet)] += 1

        if len(tweet) == 1:
            print tweet

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
        self.class_weights = {}
        for i in xrange(n_classes):
            self.class_weights[i] = (K / self.class2count[i])

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

    def init_embeddings(self, embeddings_file, emb_dim):
        unicode_chars = None
        unicode_embs = None
        dim = emb_dim
        if embeddings_file is not None:
            unicode_chars, unicode_embs = pickle.load(open(embeddings_file, "rb"))
            unicode_char2idx = {v:k for k, v in enumerate(unicode_chars)}
            dim = len(unicode_embs[0])
        W = np.zeros((len(self.char2idx) + 1, dim))
        for ch in self.char2idx:
            if unicode_chars is not None and ch in unicode_chars:
                W[self.char2idx[ch]] = unicode_embs[unicode_char2idx[ch]]
            else:
                W[self.char2idx[ch]] = np.random.uniform(-0.25, 0.25, dim)
        W[0] = np.zeros(dim, dtype = 'float32')
        return W

    def print_stats(self):
        print 'Number of unique characters: ', len(self.char2idx) + 1
        print 'Number of classes: ', len(self.label2idx)
        print 'Length distribution: ', self.len_dict
        print 'Class distribution: ', self.class2count

def main(args):

    print 'Normalize Tweets: ', args['normalize']
    time_stamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    stop_chars = read_stop_chars(args['stop_chars_file'])
    tweet_preprocessor = TweetPreprocessor(stop_chars, time_stamp, max_len = 150, word_level = args['word_level'], normalize = args['normalize'], add_ss_markers = args['add_ss_markers'])
    print 'Processing training set . . .'
    tweet_preprocessor.read_data(args['output_file_dir'], args['train_file'], parse_line, 'CONTENT', 'LABEL', is_train = True)
    print 'Processing validation set . . .'
    tweet_preprocessor.read_data(args['output_file_dir'], args['val_file'], parse_line, 'CONTENT', 'LABEL')
    print 'Processing test set . . .'
    tweet_preprocessor.read_data(args['output_file_dir'], args['test_file'], parse_line, 'CONTENT', 'LABEL')
    print 'Processing unlabeled set . . .'
    tweet_preprocessor.read_data(args['output_file_dir'], args['tweets_file'], parse_line, 'CONTENT', '')
    tweet_preprocessor.print_stats()
    tweet_preprocessor.get_class_weights()
    W = tweet_preprocessor.init_embeddings(args['embeddings_file'], args['emb_dim'])
    tweet_preprocessor.split_unlabeled_data(args['output_file_dir'], args['tweets_file'], split_ratio = 0.2)
    pickle.dump([W, tweet_preprocessor.char2idx, tweet_preprocessor.label2idx, tweet_preprocessor.class_weights, tweet_preprocessor.max_len], open(os.path.join(args['output_file_dir'], 'dictionaries_' + time_stamp + '.p'), "wb"))

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = '')
    parser.add_argument('train_file', type = str, help = 'labeled train set')
    parser.add_argument('val_file', type = str, help = 'labeled validation set')
    parser.add_argument('test_file', type = str, help = 'labeled test set')
    parser.add_argument('output_file_dir', type = str, help = 'directory where output files should be saved')

    parser.add_argument('--stop_chars_file', type = str, default = None, help = 'file containing stop characters/words')
    parser.add_argument('--embeddings_file', type = str, default = None, help = 'file containing pre-trained embeddings')
    parser.add_argument('--tweets_file', type = str, default = None, help = 'unlabeled tweets file')
    parser.add_argument('--normalize', type = bool, default = False, help = 'If True, the tweets will be normalized. Check "preprocess" method')
    parser.add_argument('--word_level', type = bool, default = False, help = 'If True, tweets will be processed at word level otherwise at char level')
    parser.add_argument('--add_ss_markers', type = bool, default = False, help = 'If True, start and stop markers will be added to the tweets')
    parser.add_argument('--emb_dim', type = int, default = 128, help = 'embedding dimension')

    args = vars(parser.parse_args())

    main(args)

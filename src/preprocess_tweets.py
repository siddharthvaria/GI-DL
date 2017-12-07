import argparse
from cStringIO import StringIO
import codecs
from collections import defaultdict
import datetime
from gensim.models import KeyedVectors
from nltk.tokenize import TweetTokenizer
import os
from random import randint
import string, re

import cPickle as pickle
from data_utils.utils import unicode_csv_reader2
import numpy as np

# from data_utils.preprocess import preprocess
aggress = set(['aggress', 'insult', 'snitch', 'threat', 'brag', 'aod', 'aware', 'authority', 'trust', 'fight', 'pride', 'power', 'lyric'])
loss = set(['loss', 'grief', 'death', 'sad', 'alone', 'reac', 'guns'])
# string.punctuation : '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
# regex_punc = re.compile('[%s]' % re.escape(string.punctuation))
regex_punc = re.compile('[\\!\\"\\$\\%\\&\\\'\\(\\)\\*\\+\\,\\-\\.\\/\\:\\;\\<\\=\\>\\?\\@\\[\\\\\\]\\^\\`\\{\\|\\}\\~]')
# regex_digit = re.compile(r"[+-]?\d+(?:\.\d+)?")

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

def preprocess(tweet, is_word_level = False):
    # replace all other white space with a single space
    tweet = re.sub('\s+', ' ', tweet)
    # remove emoji placeholders
    tweet = re.sub('(::emoji::)', '', tweet)
    # replace &amp; with and
    tweet = re.sub('&amp;', 'and', tweet)
    if is_word_level:
        tknzr = TweetTokenizer()
        tokens = tknzr.tokenize(tweet)
        tweet = ' '.join(tokens)
    # replace user handles with a constant
    tweet = re.sub('@[0-9a-zA-Z_]+', '__USER_HANDLE__', tweet)
    # replace urls
    tweet = re.sub('https?://[a-zA-Z0-9_\./]*', '__URL__', tweet)
    # replace retweet markers
    tweet = re.sub('^RT', '__RT__', tweet)
    # remove words containing digits
    tweet = re.sub(r'#*\w*\d+(?:[\./:,\-]\d+)?\w*', '', tweet).strip()
    tweet = regex_punc.sub('', tweet)
    # remove extra white space due to above operations
    tweet = re.sub(' +', ' ', tweet)
    return tweet

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
        line[text_column] = preprocess(line[text_column], is_word_level = word_level)

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
        self.wsp = ''
        if word_level:
            self.wsp = ' '
        self.normalize = normalize
        self.add_ss_markers = add_ss_markers
        self.token2idx = defaultdict(int)

        # add start and stop markers
        if self.add_ss_markers:
            self.token2idx['<'] = 1
            self.token2idx['>'] = 2

        self.label2idx = defaultdict(int)
        self.len_dict = defaultdict(int)
        self.class2count = defaultdict(int)

    def datum_to_string(self, X_ids, y_id):

        file_str = StringIO()
        file_str.write(','.join(X_ids).strip())
        file_str.write('<:>')
        file_str.write(y_id)
        return file_str.getvalue()

    def read_data(self, output_files_dir, data_file, parser, text_column, label_column, encoding, is_train = False):

        if data_file is None:
            return

        _, fname = os.path.split(data_file)
        dot_index = fname.rindex('.')
        fname_wo_ext = fname[:dot_index]

        indices_file = os.path.join(output_files_dir, fname_wo_ext + '_' + self.time_stamp + '.txt')
        new_data_file = os.path.join(output_files_dir, fname_wo_ext + '_pp_.txt')

        delimiter = get_delimiter(data_file)
        line_count = 0
        with open(data_file, 'r') as fhr, open(indices_file, 'w') as fhw1, codecs.open(new_data_file, 'w', encoding = encoding) as fhw2:
            reader = unicode_csv_reader2(fhr, encoding, delimiter = delimiter)
            for row in reader:
                line_count += 1
                # print line_count
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
                fhw2.write(self.wsp.join(X_c))
                fhw2.write('\n')
                fhw2.write('###########################################################################')
                fhw2.write('\n')
                X_ids = self.update_token2idx(X_c)
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

    def update_token2idx(self, tweet):

        if tweet is None:
            return None

        idx = []

        self.len_dict[len(tweet)] += 1

#         if len(tweet) <= 2:
#             print tweet.encode('utf8')

        if self.word_level:
            for char in tweet:
                if char not in self.token2idx:
                    self.token2idx[char] = len(self.token2idx) + 1
                idx.append(str(self.token2idx[char]))
        else:
            index = 0
            while index < len(tweet):
                if tweet[index:index + len('__URL__')] == '__URL__':
                    if '__URL__' not in self.token2idx:
                        self.token2idx['__URL__'] = len(self.token2idx) + 1
                    idx.append(str(self.token2idx['__URL__']))
                    index = index + len('__URL__')
                elif tweet[index:index + len('__USER_HANDLE__')] == '__USER_HANDLE__':
                    if '__USER_HANDLE__' not in self.token2idx:
                        self.token2idx['__USER_HANDLE__'] = len(self.token2idx) + 1
                    idx.append(str(self.token2idx['__USER_HANDLE__']))
                    index = index + len('__USER_HANDLE__')
                elif tweet[index:index + len('__RT__')] == '__RT__':
                    if '__RT__' not in self.token2idx:
                        self.token2idx['__RT__'] = len(self.token2idx) + 1
                    idx.append(str(self.token2idx['__RT__']))
                    index = index + len('__RT__')
                else:
                    ch = tweet[index]
                    if ch not in self.token2idx:
                        self.token2idx[ch] = len(self.token2idx) + 1
                    idx.append(str(self.token2idx[ch]))
                    index += 1

        return idx

    def update_label2idx(self, label):

        if label is None:
            return None

        if label not in self.label2idx:
            self.label2idx[label] = len(self.label2idx)

        return str(self.label2idx[label])

    def get_class_weights(self):

        self.class_weights = {}

        total = 0.0
        for cls in self.class2count.keys():
            total += (float(1) / self.class2count[cls])

        if total <= 0.0:
            return

        K = float(1) / total

        n_classes = len(self.label2idx)
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

    def get_onehot_vectors(self):
        W = np.zeros((len(self.token2idx) + 1, len(self.token2idx) + 1))
        for ii in xrange(len(W)):
            W[ii][ii] = 1
        return W

    def load_w2v(self, embedding_file):
        word2vec = KeyedVectors.load_word2vec_format(embedding_file, binary = True)
        print('Found %s word vectors of word2vec' % len(word2vec.vocab))
        return word2vec

    def get_dense_embeddings(self, w2v_file = None, emoji_file = None, emb_dim = 300):

        # TODO: add an option to also load pre-trained emoji embeddings
        w2v = None
        if w2v_file is not None:
            w2v = self.load_w2v(w2v_file)

        unicode_tokens = None
        unicode_embs = None
        if emoji_file is not None:
            unicode_tokens, unicode_embs = pickle.load(open(emoji_file, "rb"))
            unicode_token2idx = {v:k for k, v in enumerate(unicode_tokens)}

        W = np.zeros((len(self.token2idx) + 1, emb_dim))

        w2v_miss_count = 0
        emoji_miss_count = 0
        for token in self.token2idx:
            if w2v is not None and token in w2v.vocab:
                W[self.token2idx[token]] = w2v.word_vec(token)
            elif unicode_tokens is not None and token in unicode_tokens:
                W[self.token2idx[token]] = unicode_embs[unicode_token2idx[token]]
                w2v_miss_count += 1
            else:
                w2v_miss_count += 1
                emoji_miss_count += 1
                W[self.token2idx[token]] = np.random.uniform(-0.25, 0.25, emb_dim)

        W[0] = np.zeros(emb_dim, dtype = 'float32')

        print 'Number of tokens in vocabulary:', len(self.token2idx.keys())
        print 'Number of token embeddings found:', (len(self.token2idx.keys()) - w2v_miss_count)
        print 'Number of emoji embeddings found:', (w2v_miss_count - emoji_miss_count)
        return W

    def print_stats(self):
        print 'Number of classes: ', len(self.label2idx)
        print 'Length distribution: ', self.len_dict
        print 'Train set class distribution: ', self.class2count
        print 'Vocabulary:'
        for w in sorted(self.token2idx.keys()):
            print w.encode('utf8')

def print_args(args):

    for k, v in args.iteritems():
        print k, v

def main(args):

    print_args(args)
    time_stamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    stop_chars = read_stop_chars(args['stop_chars_file'])
    if args['word_level']:
        max_len = 53
    else:
        max_len = 150

    tweet_preprocessor = TweetPreprocessor(stop_chars, time_stamp, max_len = max_len, word_level = args['word_level'], normalize = args['normalize'], add_ss_markers = args['add_ss_markers'])
    if args['train_file'] is not None:
        print 'Processing training set . . .'
        tweet_preprocessor.read_data(args['output_file_dir'], args['train_file'], parse_line, 'text', 'label', 'utf8', is_train = True)
    if args['val_file'] is not None:
        print 'Processing validation set . . .'
        tweet_preprocessor.read_data(args['output_file_dir'], args['val_file'], parse_line, 'text', 'label', 'utf8')
    if args['test_file'] is not None:
        print 'Processing test set . . .'
        tweet_preprocessor.read_data(args['output_file_dir'], args['test_file'], parse_line, 'text', 'label', 'utf8')
    if args['tweets_file_tr'] is not None:
        print 'Processing unlabeled train set . . .'
        tweet_preprocessor.read_data(args['output_file_dir'], args['tweets_file_tr'], parse_line, 'text', 'label', 'utf8')
    if args['tweets_file_val'] is not None:
        print 'Processing unlabeled validation set . . .'
        tweet_preprocessor.read_data(args['output_file_dir'], args['tweets_file_val'], parse_line, 'text', 'label', 'utf8')
    tweet_preprocessor.get_class_weights()
    if args['use_one_hot']:
        W = tweet_preprocessor.get_onehot_vectors()
    else:
        W = tweet_preprocessor.get_dense_embeddings(args['w2v_file'], args['emoji_file'], args['emb_dim'])
    # tweet_preprocessor.split_unlabeled_data(args['output_file_dir'], args['tweets_file'], split_ratio = 0.2)
    pickle.dump([W, tweet_preprocessor.token2idx, tweet_preprocessor.label2idx, tweet_preprocessor.class_weights, tweet_preprocessor.max_len], open(os.path.join(args['output_file_dir'], 'dictionaries_' + time_stamp + '.p'), "wb"))
    tweet_preprocessor.print_stats()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = '')
    parser.add_argument('-tr', '--train_file', type = str, default = None, help = 'labeled train set')
    parser.add_argument('-val', '--val_file', type = str, default = None, help = 'labeled validation set')
    parser.add_argument('-tst', '--test_file', type = str, default = None, help = 'labeled test set')
    parser.add_argument('output_file_dir', type = str, default = None, help = 'directory where output files should be saved')

    parser.add_argument('-sch', '--stop_chars_file', type = str, default = None, help = 'file containing stop characters/words')
    parser.add_argument('-1h', '--use_one_hot', type = bool, default = False, help = 'If True, one hot vectors will be used instead of dense embeddings')
    parser.add_argument('-wfile', '--w2v_file', type = str, default = None, help = 'file containing pre-trained word2vec embeddings')
    parser.add_argument('-efile', '--emoji_file', type = str, default = None, help = 'file containing pre-trained emoji embeddings')
    parser.add_argument('-unld_tr', '--tweets_file_tr', type = str, default = None, help = 'unlabeled tweets file to be used for training')
    parser.add_argument('-unld_val', '--tweets_file_val', type = str, default = None, help = 'unlabeled tweets file to be used for validation')
    parser.add_argument('-nor', '--normalize', type = bool, default = False, help = 'If True, the tweets will be normalized. Check "preprocess" method')
    parser.add_argument('-wl', '--word_level', type = bool, default = False, help = 'If True, tweets will be processed at word level otherwise at char level')
    parser.add_argument('-amrks', '--add_ss_markers', type = bool, default = False, help = 'If True, start and stop markers will be added to the tweets')
    parser.add_argument('-edim', '--emb_dim', type = int, default = 300, help = 'embedding dimension')

    args = vars(parser.parse_args())

    main(args)

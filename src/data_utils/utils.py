from cStringIO import StringIO
import cStringIO
import codecs
import csv
from nltk.tokenize import TweetTokenizer
import os
import re


# string.punctuation : '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
# regex_punc = re.compile('[%s]' % re.escape(string.punctuation))
regex_punc = re.compile('[\\!\\"\\$\\%\\&\\\'\\(\\)\\*\\+\\,\\-\\.\\/\\:\\;\\<\\=\\>\\?\\@\\[\\\\\\]\\^\\`\\{\\|\\}\\~]')
# regex_digit = re.compile(r"[+-]?\d+(?:\.\d+)?")

def unicode_csv_reader1(utf8_data, **kwargs):
    csv_reader = csv.reader(utf8_data, **kwargs)
    for row in csv_reader:
        yield [unicode(cell, 'utf-8') for cell in row]

# To represent a unicode string as a string of bytes is known as encoding
# To convert a string of bytes to a unicode string is known as decoding.
# Use unicode('...', encoding) or '...'.decode(encoding)
def unicode_csv_reader2(utf8_data, encoding, **kwargs):
    csv_reader = csv.DictReader(utf8_data, **kwargs)
    for row in csv_reader:
        yield {k:unicode(v, encoding) for k, v in row.iteritems()}

def test_unicode_csv_reader():
    filename = '../data/csv_utf8_test.csv'
    reader = unicode_csv_reader2(open(filename))
    for line in reader:
        print len(line)
        print line

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

def get_delimiter(data_file):
    if data_file.endswith('.csv'):
        delimiter = ','
    elif data_file.endswith('.tsv'):
        delimiter = '\t'
    elif data_file.endswith('.txt'):
        delimiter = ' '
    return delimiter

def parse_line(line, text_column, label_column, tweet_id_column, max_len, stop_chars = None, normalize = False, add_ss_markers = False, word_level = False):

    if add_ss_markers:
        # -4 to account for start and stop markers and spaces
        _max_len = max_len - 4
    else:
        _max_len = max_len

    # take line (dict) as input and return text along with label if label is present
    if line[text_column] in (None, ''):
        return None, None, line[tweet_id_column]

    if stop_chars is not None:
        line[text_column] = [ch for ch in line[text_column] if ch not in stop_chars]
        line[text_column] = ''.join(line[text_column])

    if normalize:
        line[text_column] = preprocess(line[text_column], is_word_level = word_level)

    # print line[text_column]

    if line[text_column] in (None, ''):
        return None, None, line[tweet_id_column]

    if word_level:
        line[text_column] = line[text_column].split(' ')

    if len(line[text_column]) > _max_len:
        line[text_column] = line[text_column][:_max_len]

    if add_ss_markers:
        line[text_column] = '< ' + line[text_column] + ' >'

    X_c = line[text_column]
    if label_column in line.keys():
        y_c = line[label_column]
    else:
        y_c = None

    return X_c, y_c, line[tweet_id_column]

def datum_to_string(X_ids, y_id, tweet_id):

    file_str = StringIO()
    file_str.write(','.join(X_ids).strip())
    file_str.write('<:>')
    file_str.write(y_id)
    file_str.write('<:>')
    file_str.write(tweet_id)
    return file_str.getvalue()

def delete_files(flist):
    for ifile in flist:
        try:
            os.remove(ifile)
        except OSError:
            pass

class UnicodeWriter:
    """
    A CSV writer which will write rows to CSV file "f",
    which is encoded in the given encoding.
    """

    def __init__(self, f, dialect = csv.excel, encoding = "utf8", **kwds):
        # Redirect output to a queue
        self.queue = cStringIO.StringIO()
        self.writer = csv.writer(self.queue, dialect = dialect, **kwds)
        self.stream = f
        self.encoder = codecs.getincrementalencoder(encoding)()

    def writerow(self, row):
        self.writer.writerow([s.encode("utf8") for s in row])
        # Fetch utf8 output from the queue ...
        data = self.queue.getvalue()
        data = data.decode("utf8")
        # ... and reencode it into the target encoding
        data = self.encoder.encode(data)
        # write to the target stream
        self.stream.write(data)
        # empty queue
        self.queue.truncate(0)

    def writerows(self, rows):
        for row in rows:
            self.writerow(row)

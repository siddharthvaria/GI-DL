# https://raw.githubusercontent.com/bdhingra/tweet2vec/master/misc/preprocess.py

import re

regex_str = [
    r'<[^>]+>',  # HTML tags
    r'(?:@[\w_]+)',  # @-mentions
    r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)",  # hash-tags
    r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+',  # URLs

    r'(?:(?:\d+,?)+(?:\.?\d+)?)',  # numbers
    r"(?:[a-z][a-z'\-_]+[a-z])",  # words with - and '
    r'(?:[\w_]+)',  # other words
    r'(?:\S)+'  # anything else
]

tokens_re = re.compile(r'(' + '|'.join(regex_str) + ')', re.VERBOSE | re.IGNORECASE)

def tokenize(s):
    return tokens_re.findall(s)

def preprocess(s, lowercase = True):

    # put a space after qoute because otherwise "@user instances will not be handled below
    s = re.sub('"', '" ', s)
    # replace ::emoji:: with empty string
    s = re.sub('::emoji::', '', s)
    tokens = tokenize(s)
    tokens = [token.lower() for token in tokens]

    html_regex = re.compile('<[^>]+>')
    tokens = [token for token in tokens if not html_regex.match(token)]

    mention_regex = re.compile('(?:@[\w_]+)')
    tokens = ['@user' if mention_regex.match(token) else token for token in tokens]

    url_regex = re.compile('http[s]?://(?:[a-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+')
    tokens = ['!url' if url_regex.match(token) else token for token in tokens]

    hashtag_regex = re.compile("(?:\#+[\w_]+[\w\'_\-]*[\w_]+)")
    tokens = ['#hashtag' if hashtag_regex.match(token) else token for token in tokens]

#     flag = False
#     for item in tokens:
#         if item == 'rt':
#             flag = True
#             continue
#         if flag and item == '@user':
#             return ''
#         else:
#             flag = False

    return ' '.join([t for t in tokens if t]).replace('rt @user : ', '')

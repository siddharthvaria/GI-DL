import pickle
import numpy as np
import emoji

import operator

vocab = pickle.load(open('nlm/counts.pkl', 'r'))

# PREPROCESSING

def img_tweets(f):
    lines = open(f, 'r').readlines()

    tweets = []
    
    for line in lines:
        if contains_url(line):
            tweets.append(line.split()[-1].strip('\n'))

    return tweets

def non_img_tweets(f):
    lines = open(f, 'r').readlines()

    tweets = []

    for line in lines:
        if not contains_url(line):
            tweets.append(line.split('\t')[-1].strip('\n'))

    return tweets

def img_separated_tweets(f):
    lines = open(f, 'r').readlines()

    text = []
    img = []

    for line in lines:
        if contains_url(line):
#            print line
#            print line.split('\t')[-1]
            
            itweet = line.split('\t')[-1].split('http')[0]
#            if len(itweet) == 0:
#                print line
            
            img.append(itweet)
        else:
            text.append(line.split('\t')[-1])

#    print img
            
    return text, img

def tweets(txt, img):
    text = []
    image = []

    lines = open(txt, 'r').readlines()

    for line in lines:
        tw = line.split('\t')[-1]
        if not contains_url(tw):
            text.append(line.split('\t')[-1].strip('\n'))

    lines = open(img, 'r').readlines()

    for line in lines:
        splits = line.split(',')

        if len(splits) < 12:
#            print splits
            pass

        if len(splits) < 5:
            continue
        
        image.append(line.split(',')[5].strip('"'))

    return text, image

def contains_url(line):
    return 'http://t.co' in line or 'https://t.co' in line

# STATS

def compare_img_stats(txt, img):
    text, image = tweets(txt, img)

    return compare_stats(text, image)

def compare_stats(a, b):
    '''
       Length, percent emojis per tweet, empty tweets, tweets less than three
       words, only RTs, only tags, tweets more than ten words, top 15 most
       common words
    '''
    a_stats = [0, 0, 0, 0, 0, 0, 0, []]

    a_counts = {}
    
    for tweet in a:
        s = stats(tweet)

#        print s

        for i in range(7):
            a_stats[i] += s[i]

        cts = s[-1]
        
#        for word in cts.keys():
#            if word not in a_counts.keys():
#                a_counts[word] = cts[word]
#            else:
#                a_counts[word] += cts[word]

    b_stats = [0, 0, 0, 0, 0, 0, 0, []]

    b_counts = {}

    empty = 0
    
    for tweet in b:
        if len(tweet) == 0:
            empty += 1
            continue
        
        s = stats(tweet)

#        print s

        for i in range(7):
            b_stats[i] += s[i]

        cts = s[-1]

#        for word in cts.keys():
#            if word not in b_counts.keys():
#                b_counts[word] = cts[word]
#            else:
#                b_counts[word] += cts[word]

    a_stats[-1] = sorted(a_counts.items(), key=operator.itemgetter(1))[:10]
    b_stats[-1] = sorted(a_counts.items(), key=operator.itemgetter(1))[:10]

    for i in range(7):
        a_stats[i] = a_stats[i]/float(len(a))
        b_stats[i] = b_stats[i]/float(len(b))

    print 'Length, % emojis, empty, < 3 words, RTs, tags, > 10 words'

    return a_stats, b_stats, empty

def stats(tweet):
    '''
       Length, percent emojis per tweet, empty tweets, tweets less than three
       words, only RTs, only tags, tweets more than ten words, top 15 most
       common words
    '''
    counts = {}

    words = tweet.split()

    if words[-1].startswith('http'):
        tweet = ' '.join(words[:-1])
        words = words[:-1]

#    for word in words:
#        if not word in counts.keys():
#            counts[word] = 0
#        else:
#            counts[word] += 1

    
    stats = []
    stats.append(len(tweet))
    stats.append(n_emojis(tweet)/float(len(tweet)) if len(tweet) > 0 else 0)
    stats.append(1 if tweet.strip() == '' else 0)
    stats.append(1 if len(words) < 3 else 0)
    stats.append(1 if len(words) == 2 and words[1] == 'RT' else 0)
    stats.append(1 if len(words) == 1 and words[0].startswith('@') else 0)
    stats.append(1 if len(words) > 10 else 0)
    stats.append(counts)
    
    return stats

def n_emojis(tweet):
    n = 0.

    try:
        tweet = unicode(tweet, 'utf-8')
        tweet = tweet.encode('unicode_escape')
    except:
        return 0
    
#    for c in tweet:
#        if c in emoji.UNICODE_EMOJI:
#            n += 1.

#    print tweet
    splits = tweet.split('\\u')

    es = []

#    print splits

    for split in splits[1:]:
        es.append(split[:5])

    for e in es:
#        emoji = e + '\U000' + etext
#        etext = '\U000' + e
        etext = '\\u' + e

        try:
            if etext.decode('unicode_escape') in emoji.UNICODE_EMOJI:
                n += 1
        except:
            print etext

    return n

print compare_img_stats('../tsv/all.tsv', '../tsv/img.csv')

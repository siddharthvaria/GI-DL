import csv
import re
from datetime import *
from time import *
from threading import Timer
import cpd_data
from tweepy import OAuthHandler
import tweepy

import sys
sys.path.append('code/tools')
sys.path.append('code/classification')
from features import get_feats
from tools import write_out_data
from cascade import cascade

ckey = '6pobfVELAEwSIuDRJ6faOVnmN'
consumer_secret = '2jlEuzEUhTcvU7Pj9PoMnP39n2XFUTtMBfbsnpY5YdGBUw1uZa'
access_token_key = '756230494422007808-vTCg4PUiqPzK2t2z6IS57dMQykBOsa7'
access_token_secret = 'niBkq757O84aIYTcF846P2fbySKrIpDYu9FAO7jsaSWRE'


def correlate(last_seen_id=None, num_days=5):
    #set up with today's date
    today = date.today()
    print today #TESTING
    x_days_ago = today-timedelta(days=num_days)

    #pull data from cpd from past month
    cpd_incidents = cpd_data.get_cpd_data(today)
     #TESTING
    for c in cpd_incidents:
        print c
        print

    #go through tweets from past week
    reader = csv.DictReader(open('/proj/nlp/users/terra/streaming_results/twitter_stream_result.csv', 'rb'))
    if last_seen_id: too_far_back=True
    else: too_far_back=False

    recent_tweets = []
    usernames = []
    most_recent_id = None
    for row in reader:
        if too_far_back:
            if row['ID'] == last_seen_id: too_far_back=False
            continue
       
        ts = strptime(row['DATE'], '%a %b %d %H:%M:%S +0000 %Y')
        dt = date.fromtimestamp(mktime(ts))
        
        if dt > x_days_ago:
            break

        #else: 
        most_recent_id = row['ID']
        recent_tweets.append((row['ID'], row['CONTENT'], dt))

        #get the usernames from tweet
        usernames.extend(get_usernames(row['CONTENT']))
        
    #if labeled aggressive, put into aggressive list
    #aggressive_tweets = [(id, date)]
    aggressive_tweets = get_aggressive_tweets(recent_tweets)    

    #TODO correlate events within 5 days after each aggress tweet with resp. tweet and output correlations to file with date
    correlated_data = {}
    for tweet in aggressive_tweets:
        tweet_id = tweet[0]
        tweet_date = tweet[1]
        end_date = tweet_date+timedelta(days=num_days)
        related_incidents = []
        
        for c in cpd_incidents:
            ts = strptime(c['date'], '%Y-%m-%dT%H:%M:%S.000')
            event_date = date.fromtimestamp(mktime(ts))
            if event_date >= tweet_date and event_date < end_date: 
                related_incidents.append(c)
        
        correlated_data[tweet_id] = related_incidents

    #appends this week's data to file
    write_out_correlations(correlated_data)

    #writes username and userid to file if user description has gang hashtags
    write_out_usernames(usernames)
    print 'correlated!'
    return most_recent_id

def get_usernames(tweet):
    twitter_username_re = re.compile(r'@([A-Za-z0-9_]+)')
    results = []
    for m in twitter_username_re.finditer(tweet):
        results.append(m)
    return results

#for now just writing out to (an additional) file. TODO automatically verify
def write_out_usernames(names):
    #auth is a global variable
    api = tweepy.API(auth)

    e = open('/proj/nlp/users/terra/data/usernames.txt', 'r')
    f = open('/proj/nlp/users/terra/streaming_results/gathered_usernames.txt', 'a')

    current_ids = []
    for line in e.readlines():
        current_ids.append(line.strip().split()[1])

    for name in names:
        try:
            u = api.get_user(name)
            if u.id_str not in current_ids: f.write(name+' '+str(u.id_str)+'\n')
            current_ids.append(u.id_str)
        except: 
            continue

    f.close()
    e.close()
    return

def write_out_correlations(data):
    f = open('/proj/nlp/users/terra/streaming_results/correlation.txt', 'a')

    for key in data.keys():
        events = data[key]
        f.write(key+'\n')
        for e in events:
            f.write(e+'\n')
        if len(events) == 0: f.write('no relevant events\n')

        f.write('\n')

    f.close()
    return

def get_aggressive_tweets(tweets):
    content = [t[1] for t in tweets]
    id_date = [(t[0], t[2]) for t in tweets]
    results = []


    write_out_data(tweets, '/proj/nlp/users/terra/streaming_results/correlate_tmp.csv')
    predictions = cascade('/proj/nlp/users/terra/streaming_results/correlate_tmp.csv')

    for p in range(0, len(predictions)):
        if predictions[p] == 1:
            results.append(id_date[p])

    return results



#API connection
auth = OAuthHandler(ckey, consumer_secret)
auth.set_access_token(access_token_key, access_token_secret)

tweet_id = None
while 1:
    tweet_id = correlate(tweet_id, 5)
    print tweet_id
    #TODO - set timer for next week
    delta_t=timedelta(minutes=1)
    secs=delta_t.total_seconds()
    print secs
    sleep(secs)
    

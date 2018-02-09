"""
Source: http://stats.seandolinar.com/collecting-twitter-data-using-a-python-stream-listener/
"""
 
import time
from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
#from pymongo import MongoClient
import json
import csv
import subprocess

ckey = '6pobfVELAEwSIuDRJ6faOVnmN'
consumer_secret = '2jlEuzEUhTcvU7Pj9PoMnP39n2XFUTtMBfbsnpY5YdGBUw1uZa'
access_token_key = '756230494422007808-vTCg4PUiqPzK2t2z6IS57dMQykBOsa7'
access_token_secret = 'niBkq757O84aIYTcF846P2fbySKrIpDYu9FAO7jsaSWRE'
 
start_time = time.time()  # grabs the system time
username_file = '/proj/nlp/users/terra/data/usernames.txt'
 
#got user ids for each username at http://mytwitterid.com/ 
def read_in_users(filename):
    usernames = open(filename)
    result = []
    for name in usernames.readlines():
        user_id = name.strip().split(' ')[1]
        if int(user_id) > 0 : result.append(user_id)
    print result
    return result

def tokenize_tweet(text):
    proc = subprocess.Popen(['bash', '/proj/nlp/users/terra/gang-intervention/tweet_preprocessing/individual_tweet_tokenize.sh'], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    tokenized = proc.communicate(text.rstrip().encode('utf-8'))[0]
    tokenized = tokenized.rstrip().split('\n')[-1]
    print tokenized #TESTING
    return tokenized

# Listener Class Override
 
class listener(StreamListener):
    def __init__(self, start_time):
        self.time = start_time
        self.num_tweets = 0
        self.retrieved_tweets = []


    def on_data(self, data):
        f = open('/proj/nlp/users/terra/streaming_results/twitter_stream_result.json', 'a')
	writer = csv.DictWriter(open('/proj/nlp/users/terra/streaming_results/twitter_stream_result.csv', 'ab'), ['ID', 'AUTHOR', 'CONTENT', 'DATE'])

        while 1:
            try:
                tweet = json.loads(data)
                tweetdata = {
                    'id': tweet['id_str'],
                    'text': tweet['text'],
                    'created_at': tweet['created_at'],
                    'user': {
                        'id': tweet['user']['id_str'],
                        'screen_name': tweet['user']['screen_name'],
                        'name': tweet['user']['name'],
                        'description': tweet['user']['description'],
                        'location': tweet['user']['location'],
                        'created_at': tweet['user']['created_at'],
                        'favourites_count': tweet['user']['favourites_count'],
                        'statuses_count': tweet['user']['statuses_count'],
                        'friends_count': tweet['user']['friends_count'],
                        'time_zone': tweet['user']['time_zone'],
                        'utc_offset': tweet['user']['utc_offset'],
                        'lang': tweet['user']['lang'],
                        'followers_count': tweet['user']['followers_count']
                    },
                    'geo': tweet['geo'],
                    'coordinates': tweet['coordinates'],
                    'place': tweet['place'],
                    'source': tweet['source']
                }
                if tweetdata['id'] not in self.retrieved_tweets:
                    t = tokenize_tweet(tweetdata['text'])
                    tweetdata['text'] = t
                    
                    f.write("{}\n".format(json.dumps(tweetdata)))
                    self.retrieved_tweets.append(tweetdata['id'])
                    self.num_tweets += 1
                    print 'got '+str(self.num_tweets)+' tweets'

                    print 'tweetdata text = '+str(type(tweetdata['text']))
                    d = {'ID': tweetdata['id'], 'AUTHOR': tweetdata['user']['screen_name'], 'CONTENT': tweetdata['text'], 'DATE': tweetdata['created_at']}
                    writer.writerow(d)
        
                #refresh the connection every 24 hours in case of updates to the userlist
                if time.time() - self.time > 86400:
                    self.time = time.time()
                    return False
                else:
                    return True 
     
            except BaseException, e:
                print 'failed ondata, ', e
                return True
        f.close()
        exit()
 
    def on_error(self, status):
        print str(status)
        if status == 420:
            return False
 
auth = OAuthHandler(ckey, consumer_secret)
auth.set_access_token(access_token_key, access_token_secret)
twitterStream = Stream(auth, listener(start_time))

print 'set up complete'
while 1:
    user_list = read_in_users(username_file)
    twitterStream.filter(follow=user_list)  # call the filter method to run the Stream object
    twitterStream.disconnect()

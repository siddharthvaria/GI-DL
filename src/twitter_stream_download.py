# To run this code, first edit config.py with your configuration, then:
#
# mkdir data
# python twitter_stream_download.py -q apple -d data
#
# It will produce the list of tweets for the query "apple"
# in the file data/stream_apple.json

import tweepy
from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
import time
import argparse
import string
import json

consumer_key = ''
consumer_secret = ''
access_key = ''
access_secret = ''

def get_parser():
    """Get parser for command line arguments."""
    parser = argparse.ArgumentParser(description = "Twitter Downloader")
#     parser.add_argument("-q",
#                         "--query",
#                         dest = "query",
#                         help = "Query/Filter",
#                         default = '-')
    parser.add_argument("-d",
                        "--data-dir",
                        dest = "data_dir",
                        help = "Output/Data Directory")
    return parser

class MyListener(StreamListener):
    """Custom StreamListener for streaming data."""

    def __init__(self, data_dir):
        # query_fname = format_filename(query)
        # self.outfile = "%s/stream_%s.json" % (data_dir, query_fname)
        self.outfile = "%s/stream.txt" % (data_dir)

    def on_data(self, data):
        try:
            with open(self.outfile, 'a') as f:
                json_data = json.loads(data)
                # print(json_data["text"])
                if "text" in json_data.keys():
                    f.write(json.dumps({"text": json_data["text"]}))
                    f.write('\n')
                return True
        except BaseException as e:
            print("Error on_data: %s" % str(e))
            time.sleep(5)
        return True

    def on_error(self, status):
        print(status)
        return True


def format_filename(fname):
    """Convert file name into a safe string.

    Arguments:
        fname -- the file name to convert
    Return:
        String -- converted file name
    """
    return ''.join(convert_valid(one_char) for one_char in fname)


def convert_valid(one_char):
    """Convert a character into '_' if invalid.

    Arguments:
        one_char -- the char to convert
    Return:
        Character -- converted char
    """
    valid_chars = "-_.%s%s" % (string.ascii_letters, string.digits)
    if one_char in valid_chars:
        return one_char
    else:
        return '_'

# @classmethod
# def parse(cls, api, raw):
#     status = cls.first_parse(api, raw)
#     setattr(status, 'json', json.dumps(raw))
#     return status

def main(args):

    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_secret)
    api = tweepy.API(auth)

    twitter_stream = Stream(auth, MyListener(args.data_dir))
    # twitter_stream.filter(['448734451'])
    twitter_stream.filter(track = ['Trump'])

if __name__ == '__main__':

    parser = get_parser()

    args = parser.parse_args()

    main(args)

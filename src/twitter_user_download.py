#!/usr/bin/env python
# encoding: utf-8
import tweepy  # https://github.com/tweepy/tweepy
import csv
import configparser
import argparse
import os
import time

def get_api(configFilePath):
    config = configparser.ConfigParser()
    config.read(configFilePath)
    auth = tweepy.OAuthHandler(config['twitter.com']['consumer_key'], config['twitter.com']['consumer_secret'])
    # auth.set_access_token(access_key, access_secret)
    auth.set_access_token(config['twitter.com']['access_key'], config['twitter.com']['access_secret'])
    api = tweepy.API(auth)
    return api

def get_all_tweets(api, screen_name):
    # Twitter only allows access to a users most recent 3240 tweets with this method

    # initialize a list to hold all the tweepy Tweets
    alltweets = []

    # make initial request for most recent tweets (200 is the maximum allowed count)
    new_tweets = []
    try:
        new_tweets = api.user_timeline(screen_name = screen_name, count = 200)
    except tweepy.TweepError:
        if tweepy.TweepError is "[{'code': 130, 'message': 'Over capacity'}]":
            print('Got error code: Over capacity')
            time.sleep(2)

    # save most recent tweets
    alltweets.extend(new_tweets)

    if len(alltweets) == 0:
        return None

    # save the id of the oldest tweet less one

    oldest = alltweets[-1].id - 1

    # keep grabbing tweets until there are no tweets left to grab
    while len(new_tweets) > 0:
        # print ("getting tweets before %s" % (oldest))

        # all subsiquent requests use the max_id param to prevent duplicates
        # new_tweets = api.user_timeline(screen_name = screen_name, count = 200, max_id = oldest)

        new_tweets = []
        try:
            new_tweets = api.user_timeline(screen_name = screen_name, count = 200, max_id = oldest)
        except tweepy.TweepError:
            if tweepy.TweepError is "[{'code': 130, 'message': 'Over capacity'}]":
                print('Got error code: Over capacity')
                time.sleep(2)

        # save most recent tweets
        alltweets.extend(new_tweets)

        # update the id of the oldest tweet less one
        oldest = alltweets[-1].id - 1

        # print ("...%s tweets downloaded so far" % (len(alltweets)))

    # transform the tweepy tweets into a 2D array that will populate the csv
    outtweets = [[screen_name, tweet.id_str, tweet.created_at, tweet.text] for tweet in alltweets]

    return outtweets

def get_users(users_file):

    users = []
    with open(users_file, 'r') as fh:
        for line in fh:
            line = line.strip()
            users.append(line)

    return users

def main(args):

    api = get_api(args['credentials_file'])
    users = get_users(args['users_file'])

    with open(os.path.join(args['output_dir'], 'tweets.csv'), 'w') as f:
        writer = csv.writer(f)
        writer.writerow(["user_name", "tweet_id", "created_at", "text"])
        user_count = 1
        for user in users:
            if user_count % 1 == 0:
                print('Processing user: %s (%d/%d)' % (user, user_count, len(users)))
            outtweets = get_all_tweets(api, user)
            if outtweets is not None:
                writer.writerows(outtweets)
            user_count += 1

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = '')
    parser.add_argument('credentials_file', type = str)
    parser.add_argument('users_file', type = str)
    parser.add_argument('output_dir', type = str)
    args = vars(parser.parse_args())
    main(args)

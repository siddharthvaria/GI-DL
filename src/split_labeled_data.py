import argparse
from collections import defaultdict
import math
import os
from sklearn.model_selection import StratifiedShuffleSplit

from data_utils.utils import UnicodeWriter
from data_utils.utils import unicode_csv_reader2
import numpy as np


def split(data, schema, ratio):
    tr_data = defaultdict(list)
    te_data = defaultdict(list)
    sss = StratifiedShuffleSplit(n_splits = 1, test_size = ratio, random_state = 123)
    for train_index, test_index in sss.split(np.zeros(len(data['label'])), data['label']):
        for column in schema:
            tr_data[column] = data[column][train_index]
            te_data[column] = data[column][test_index]
    return tr_data, te_data


def get_stratified_samples(data, schema):

    for column in schema:
        data[column] = np.asarray(data[column])

    ensemble_tweets_r = math.floor(float(1000 * 100) / len(data['label'])) / 100
    # print 'ensemble_tweets_r: ', ensemble_tweets_r
    _tr_data, ensemble_data = split(data, schema, ensemble_tweets_r)

    test_tweets_r = math.floor(float(2000 * 100) / len(_tr_data['label'])) / 100
    # print 'test_tweets_r: ', test_tweets_r
    _tr_data, te_data = split(_tr_data, schema, test_tweets_r)

    val_tweets_r = math.floor(float(1500 * 100) / len(_tr_data['label'])) / 100
    # print 'val_tweets_r: ', val_tweets_r
    tr_data, val_data = split(_tr_data, schema, val_tweets_r)

    print 'Train set size: ', len(tr_data['tweet_id'])
    print 'Val set size: ', len(val_data['tweet_id'])
    print 'Test set size: ', len(te_data['tweet_id'])
    print 'Ensemble set size: ', len(ensemble_data['tweet_id'])

    return tr_data, val_data, te_data, ensemble_data


def read_data(input_file, schema):
    data = defaultdict(list)
    tweet_ids = defaultdict(list)
    label2count = defaultdict(int)

    with open(input_file, 'r') as fhr:
        reader = unicode_csv_reader2(fhr, 'utf8', delimiter = ',')
        for row in reader:
            tweet_id = row['tweet_id']
            label = row['label']

            if label not in ['Loss', 'Aggression', 'Other']:
                print 'Found invalid label: ', label
                continue

            label2count[row['label']] += 1
            tweet_ids[tweet_id].append(label)

    duplicate_tweets = set()
    with open(input_file, 'r') as fhr:
        reader = unicode_csv_reader2(fhr, 'utf8', delimiter = ',')
        for row in reader:
            tweet_id = row['tweet_id']
            label = row['label']

            if label not in ['Loss', 'Aggression', 'Other']:
                print 'Found invalid label: ', label
                continue

            if tweet_id not in tweet_ids:
                raise Exception('How did this happen!')
            else:
                _labels = set(tweet_ids[tweet_id])
                if len(_labels) > 1:
                    print 'Disagreement found, Tweet id: %s' % (tweet_id)
                    continue

                if tweet_id in duplicate_tweets:
                    print 'Found duplicate tweet: ', tweet_id
                    continue
                else:
                    duplicate_tweets.add(tweet_id)

                for column in schema:
                    data[column].append(row[column])

    print 'Len(data): ', len(data['tweet_id'])
    print 'Label distribution: ', label2count
    return data


def write_data(data, schema, output_file):
    assert len(schema) == len(data.keys()), 'data.keys() does not match the schema!'
    with open(output_file, 'w') as fh:
        unicode_writer = UnicodeWriter(fh)
        unicode_writer.writerow(schema)
        for ii in xrange(len(data[schema[0]])):
            datum_lst = []
            for k in schema:
                datum_lst.append(data[k][ii])
            unicode_writer.writerow(datum_lst)


def main(args):
    schema = ['tweet_id', 'user_name', 'text', 'label', 'created_at']
    data = read_data(args.input_file, schema)

    tr_data, val_data, te_data, ensemble_data = get_stratified_samples(data, schema)

    fpath, fname = os.path.split(args.input_file)
    dot_index = fname.rindex('.')
    fname_wo_ext = fname[:dot_index]

    output_file = os.path.join(fpath, fname_wo_ext + '_tr.csv')
    write_data(tr_data, schema, output_file)

    output_file = os.path.join(fpath, fname_wo_ext + '_val.csv')
    write_data(val_data, schema, output_file)

    output_file = os.path.join(fpath, fname_wo_ext + '_test.csv')
    write_data(te_data, schema, output_file)

    output_file = os.path.join(fpath, fname_wo_ext + '_ensemble.csv')
    write_data(ensemble_data, schema, output_file)


def parse_args():
    parser = argparse.ArgumentParser(description = '')
    parser.add_argument('-i', '--input-file', type = str, required = True)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)

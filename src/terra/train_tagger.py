import sys
#import nltk
import math
import time
import string
import numpy as np
import gensim

import tagger as t

import pybrain
from pybrain.datasets import supervised
from pybrain.tools.shortcuts import buildNetwork

from sklearn import svm
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron

def main():
    # start timer
    time.clock()


    # Training the Tagger:

    # open training data
    #infile = open("../data/gold/simple_gold_revised_emojis.txt", "r")
    infile = open("../results/pos_tagged_4_fold_cv.txt", "r")
    t_sents = infile.readlines()
    infile.close()
    #train_sents = []
    train_sents = list(t_sents[102:])
    #train_sents = list(t_sents)
    # open CMU training data
    infile = open("../data/gold/cmu_all_gold.txt")
    sents = infile.readlines()
    cmu_train_sents = sents
    #cmu_train_sents = sents[:1328]
    #cmu_train_sents.extend(sents[1428:])
    #cmu_train_sents = []
    infile.close()

    all_sents = list()
    all_sents.extend(train_sents)
    all_sents.extend(cmu_train_sents)

    # Standard implementation of domain adaptation
    domain_list = ['tweet']*len(train_sents)
    #domain_list.extend(['tweet']*len(cmu_train_sents))
    domain_list.extend(['cmu']*len(cmu_train_sents))
    #domain_list = None


    # Initializing the tagger
    tagger = t.tagger(brown_cluster_path='../tools/TweeboParser/pretrained_models/twitter_brown_clustering_full',
                      word2vec_path='../tools/word2vec/word2vec_twitter_model.bin'

                      #word2vec_path= '../tools/word2vec/glove.6B/glove.6B.300d.txt',
                      #word2vec_path= '../tools/word2vec/glove.840B.300d/glove.840B.300d.txt'
                      #word2vec_path= '../tools/word2vec/glove.twitter.27B/glove.twitter.27B.200d.txt',
                      #word2vec_path= '../tools/word2vec/GoogleNews-vectors-negative300.bin'
                      #wiktionary_path='../data/wiktionary'
                      )


    #tagged_sents = tagger.cross_validation(train_sents, domain_list, len(train_sents), folds=4)
    #tagger.output_tagged(tagged_sents, '../results/pos_tagged_4_fold_cv.txt',)

    tagger.train(all_sents, domain_list)

    tagger.save_clf(path='../classifiers/POS-tagger.pkl')

    # Using the tagger to tag dev set data

    # open Corpus development data

    #infile = open("../data/content/simple_content_emoji.txt", "r")
    infile = open("../data/gold/simple_gold_revised_emojis.txt", "r")
    #infile = open("../data/gold/test_final.txt", "r")
    print('Reading Dev')
    train_Dev = infile.readlines()[:200]
    infile.close()
    dev_tokens, _ = tagger.preprocess(train_Dev)

    print('Testing Dev')
    tagged_sents = tagger.tag_sents(dev_tokens, 'tweet')
    print('Writing Results')
    tagger.output_tagged(tagged_sents, '../results/pos_tagged_cv.txt')


    infile = open("../data/content/test_final_content.txt", "r")
    print('Reading Dev')
    train_test = infile.readlines()[:200]
    infile.close()
    test_tokens, _ = tagger.preprocess(train_test)
    print('Testing Dev')
    tagged_sents = tagger.tag_sents(test_tokens, 'tweet')
    print('Writing Results')
    tagger.output_tagged(tagged_sents, '../results/pos_tagged_test_cv.txt')

    '''
    infile = open("../data/gold/cmu_test_gold.txt", "r")
    train_cmu = infile.readlines()
    cmu_tokens, _ = tagger.preprocess(train_cmu)
    tagged_sents = tagger.tag_sents(cmu_tokens, 'cmu')
    tagger.output_tagged(tagged_sents, '../results/cmu_pos_tagged_cv.txt')
    '''

    print("Time: " + str(time.clock()) + ' sec')

if __name__ == "__main__":
    main()

import eval_parsing

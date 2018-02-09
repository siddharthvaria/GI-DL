__author__ = 'robertk'

import tagger as t
import sys

def main():
    input_path = sys.argv[1] # "../data/content/test_final_content.txt"
    classifier_path = sys.argv[2]  # '../classifiers/POS-tagger.pkl'
    brown_path = sys.argv[3] # '../tools/TweeboParser/pretrained_models/twitter_brown_clustering_full'

    tagger = t.tagger(brown_cluster_path=brown_path)
    tagger.load_clf(classifier_path)

    infile = open(input_path, "r")
    train_sents = infile.readlines()
    infile.close()
    train_tokens, _ = tagger.preprocess(train_sents)
    tagged_sents = tagger.tag_sents(train_tokens, 'tweet')
    conll_sents = tagger.convert_conll(tagged_sents)
    tagger.output_tagged(conll_sents)


if __name__ == "__main__":
    main()
import argparse
import datetime
from sklearn.metrics import classification_report
from cnn_lm_nce import CNN_Model
from data_utils.TweetReader2 import TweetCorpus
from tensorflow_impl.train_cnn_lm_nce import binarize_labels
import numpy as np
import tensorflow as tf
from train_clf import batch_iter
from train_clf import dev_step


def test_clf(sess, model, args, corpus):
    _, _, X_test, _, _, y_test = corpus.get_data_for_classification()
    y_test = binarize_labels(y_test, args['pos_classes'], corpus.label2idx)

    saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES), max_to_keep = 1)

    print 'Initializing all variables . . .'
    sess.run(tf.global_variables_initializer())

    if args['pretrained_weights'] is not None:
        print 'Restoring weights from existing checkpoint . . .'
        saver.restore(sess, args['pretrained_weights'])

        test_batches = batch_iter(list(zip(X_test, y_test)), args['batch_size'])
        test_probabilities = []
        for batch in test_batches:
            x_batch, y_batch = zip(*batch)
            _, probabilities = dev_step(sess, model, x_batch, y_batch)
            test_probabilities.extend(probabilities)

    print classification_report(y_test, np.argmax(test_probabilities, axis = 1), target_names = corpus.get_class_names())


def main(args):

    corpus = TweetCorpus(test_file = args['test_file'], dictionaries_file = args['dictionaries_file'])

    ts = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    args['ts'] = ts

    class_weights = [corpus.class_weights[i] for i in range(len(corpus.label2idx))]

    # print_hyper_params(args)

    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
          allow_soft_placement = True,
          log_device_placement = False)
        sess = tf.Session(config = session_conf)
        with sess.as_default():
            cnn = CNN_Model(sequence_length = corpus.max_len,
                               num_classes = len(corpus.label2idx),
                               vocab_size = len(corpus.token2idx),
                               embedding_size = len(corpus.W[0]),
                               filter_sizes = [1, 2, 3, 4, 5],
                               num_filters = args['nfeature_maps'],
                               embeddings = corpus.W,
                               class_weights = class_weights)

            # sess, model, args, corpus
            test_clf(sess, cnn, args, corpus)


def parse_arguments():

    parser = argparse.ArgumentParser(description = '')
    parser.add_argument('-tst', '--test_file', type = str, help = 'labeled test file')
    parser.add_argument('-dict', '--dictionaries_file', type = str, help = 'pickled dictionary file')
    parser.add_argument('-w', '--pretrained_weights', type = str, default = None, help = 'Path to pretrained weights file')
    parser.add_argument('-nfmaps', '--nfeature_maps', type = int, default = 200)
    parser.add_argument('-bsz', '--batch_size', type = int, default = 256)
    parser.add_argument('-pc', '--pos-classes', nargs = '+', default = [])
    args = vars(parser.parse_args())

    return args


if __name__ == '__main__':

    main(parse_arguments())

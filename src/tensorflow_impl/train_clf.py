import argparse
import datetime
from numpy.random import choice
import os
import shutil
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
import time

from Nadam import NadamOptimizer
from cnn_lm_nce import CNN_Model
from data_utils.TweetReader2 import TweetCorpus
from tensorflow_impl.train_cnn_lm_nce import binarize
import numpy as np
import tensorflow as tf
import cPickle as pickle


def batch_iter(data, batch_size, shuffle = False):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
    # Shuffle the data at each epoch
    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_data = data[shuffle_indices]
    else:
        shuffled_data = data
    for batch_num in range(num_batches_per_epoch):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        yield shuffled_data[start_index:end_index]


def dev_step(sess, model, x_batch, y_batch):

    """
    Evaluates model on a dev set based on the mode
    """
    feed_dict = {
      model.input_x: x_batch,
      model.input_y_clf: y_batch,
      model.dropout_keep_prob: 1.0
      # cnn.is_train: 0
    }
    loss, probabilities = sess.run(
        [model.clf_loss, model.probabilities],
        feed_dict)
    return loss, probabilities


def train_clf(sess, model, args, corpus):

    # Define Training procedure
    global_step = tf.Variable(0, name = "global_step", trainable = False)
    # optimizer = tf.train.AdamOptimizer(learning_rate = 1e-2)
    optimizer = NadamOptimizer()
    grads_and_vars = optimizer.compute_gradients(model.clf_loss)
    train_op = optimizer.apply_gradients(grads_and_vars, global_step = global_step)
    grad_summaries = []
    for g, v in grads_and_vars:
        if g is not None:
            grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
            sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
            grad_summaries.append(grad_hist_summary)
            grad_summaries.append(sparsity_summary)
    grad_summaries_merged = tf.summary.merge(grad_summaries)

    # Output directory for models and summaries
    if os.path.isdir(os.path.join(args['model_save_dir'], "clf_runs")):
        shutil.rmtree(os.path.join(args['model_save_dir'], "clf_runs"))
    out_dir = os.path.abspath(os.path.join(args['model_save_dir'], "clf_runs"))
    print("Writing to {}\n".format(out_dir))

    # Summaries for loss and accuracy
    loss_summary = tf.summary.scalar("loss", model.clf_loss)

    # Train Summaries
    train_summary_op = tf.summary.merge([loss_summary, grad_summaries_merged])
    train_summary_dir = os.path.join(out_dir, "summaries", "train")
    train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

    # Dev summaries
#             dev_summary_op = tf.summary.merge([loss_summary])
#             dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
#             dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

    # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
    checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
    checkpoint_prefix = os.path.join(checkpoint_dir, "model")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES), max_to_keep = 1)

    print 'Initializing all variables . . .'
    sess.run(tf.global_variables_initializer())

    if args['pretrained_weights'] is not None:
        print 'Restoring weights from existing checkpoint %s' % (args['pretrained_weights'])
        saver.restore(sess, args['pretrained_weights'])

    def train_step(x_batch, y_batch):
        """
        A single training step based on the mode
        """
        feed_dict = {
          model.input_x: x_batch,
          model.input_y_clf: y_batch,
          model.dropout_keep_prob: args['dropout']
          # cnn.is_train: 1
        }
        _, step, summaries, loss, probabilities = sess.run(
            [train_op, global_step, train_summary_op, model.clf_loss, model.probabilities],
            feed_dict)
        train_summary_writer.add_summary(summaries, step)

        return loss, probabilities

    X_tr, X_val, _, y_tr, y_val, _ = corpus.get_data_for_classification()

    best_val_f = 0
    # best_val_loss = 1e9
    best_val_probabilities = None
    patience = 5

    for epoch in xrange(args['n_epochs']):
        _tr_loss = 0
        # tr_probabilities = []
        print 'Epoch %d:' % (epoch)
        _start = time.time()
        batches = batch_iter(
            list(zip(X_tr, y_tr)), args['batch_size'], shuffle = True)
        # Training loop. For each batch...
        for batch in batches:
            x_batch, y_batch = zip(*batch)
            _loss, _ = train_step(x_batch, y_batch)
            _tr_loss += _loss
            # tr_probabilities.extend(probabilities)

        current_step = tf.train.global_step(sess, global_step)
        tr_loss = _tr_loss / len(X_tr)
        print 'Run time: %d s' % (time.time() - _start)
        print 'Training Loss: %f' % (tr_loss)

        val_batches = batch_iter(list(zip(X_val, y_val)), args['batch_size'])
        _val_loss = 0
        val_probabilities = []
        for batch in val_batches:
            x_batch, y_batch = zip(*batch)
            _loss, probabilities = dev_step(sess, model, x_batch, y_batch)
            _val_loss += _loss
            val_probabilities.extend(probabilities)

        val_loss = _val_loss / len(X_val)
        val_f = f1_score(y_val, np.argmax(val_probabilities, axis = 1), average = 'macro')

        print 'Val Loss: %f, Val macro f: %f' % (val_loss, val_f)

        if val_f > best_val_f:
        # if val_loss < best_val_loss:
            best_val_f = val_f
            # best_val_loss = val_loss
            best_val_probabilities = val_probabilities
            path = saver.save(sess, checkpoint_prefix, global_step = current_step)
            print("Saved model checkpoint to {}\n".format(path))
            patience = 5
        else:
            patience -= 1
            print("\n")

        if patience == 0:
            print 'Early stopping . . .'
            break

    print classification_report(y_val, np.argmax(best_val_probabilities, axis = 1), target_names = corpus.get_class_names())
    pickle.dump([y_val, best_val_probabilities], open(os.path.join(args['model_save_dir'], 'probabilities' + '_' + args['pos_classes'][0] + '.p'), 'wb'))


def main(args):

    corpus = TweetCorpus(train_file = args['train_file'], val_file = args['val_file'], test_file = None , dictionaries_file = args['dictionaries_file'])

    if args['binary']:
        binarize(corpus, args['pos_classes'])

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
                               filter_sizes = [1, 2],
                               num_filters = args['nfeature_maps'],
                               embeddings = corpus.W,
                               class_weights = class_weights)

            print 'List of trainable variables:'
            for i in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
                print i.name

            # sess, model, args, corpus
            train_clf(sess, cnn, args, corpus)


def parse_arguments():

    parser = argparse.ArgumentParser(description = '')
    # even though short flags can be used in the command line, they can not be used to access the value of the arguments
    # i.e args['pt'] will give KeyError.
    parser.add_argument('-tr', '--train_file', type = str, help = 'labeled train file')
    parser.add_argument('-val', '--val_file', type = str, help = 'labeled validation file')
    parser.add_argument('-dict', '--dictionaries_file', type = str, help = 'pickled dictionary file')
    parser.add_argument('-sdir', '--model_save_dir', type = str, help = 'directory where trained model should be saved')
    parser.add_argument('-w', '--pretrained_weights', type = str, default = None, help = 'Path to pretrained weights file')
    parser.add_argument('-epochs', '--n_epochs', type = int, default = 30)
    parser.add_argument('-nfmaps', '--nfeature_maps', type = int, default = 300)
    parser.add_argument('-do', '--dropout', type = float, default = 0.5)
    parser.add_argument('-bsz', '--batch_size', type = int, default = 256)
    parser.add_argument('-b', '--binary', type = bool, default = False, help = 'If binary is True, then pos-classes will be used to identify the positive class.')
    parser.add_argument('-pc', '--pos-classes', nargs = '+', default = [])

    args = vars(parser.parse_args())

    return args


if __name__ == '__main__':

    main(parse_arguments())

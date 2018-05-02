import argparse
import datetime
import os
import shutil
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_selection.mutual_info_ import mutual_info_classif
import time
import torch

from Nadam import NadamOptimizer
from scipy.sparse import csc_matrix
from cnn_lm_nce import CNN_Model
from data_utils.TweetReader2 import TweetCorpus
import numpy as np
import tensorflow as tf


def get_stopwords(stop_words_file):

    stop_words = []
    with open(stop_words_file, 'r') as fh:
        for line in fh:
            stop_words.append(line.strip())

    return stop_words


def drop_tweets_by_length(inputs, targets, min_len = 5):

    # drop tweets less than min_len long
    inputs_n = []
    targets_n = []
    for x, t in zip(inputs, targets):
        if len(x) < min_len:
            continue
        inputs_n.append(x)
        targets_n.append(t)
    inputs_n = np.asarray(inputs_n)
    targets_n = np.asarray(targets_n)
    return inputs_n, targets_n


def convert_X_to_ijv_format(inputs):
    # duplicates are fine as they are handled by 'CSC' format.
    data = []
    rows = []
    columns = []
    for ii in xrange(inputs.shape[0]):
        for idx in inputs[ii]:
            data.append(1.0)
            rows.append(ii)
            columns.append(idx)

    return data, rows, columns


'''
def convert_y_to_ijv_format(y):

    data = []
    rows = []
    columns = []
    for ii in xrange(y.shape[0]):
        data.append(y[ii])
        rows.append(ii)
        columns.append(0)

    return data, rows, columns
'''


def get_correlations(inputs, targets, idx2token, stop_words = None, mask_value = None):

    print 'Before dropping, len(inputs): ', len(inputs)
    inputs, targets = drop_tweets_by_length(inputs, targets)
    print 'After dropping, len(inputs): ', len(inputs)

    # convert X to CSC format
    data, row, col = convert_X_to_ijv_format(inputs)
    counts = csc_matrix((data, (row, col)), shape = (inputs.shape[0], len(idx2token)))

    tf_idf_transformer = TfidfTransformer(norm = 'l2', use_idf = True, smooth_idf = True, sublinear_tf = True)
    tf_idf_transformer.fit(counts)
    counts = tf_idf_transformer.transform(counts)

    zero_cnt_words = 0
    mx_corrcoef = 0.0
    mn_corrcoef = 0.0

    corrcoef = np.zeros(len(idx2token))

    print 'Computing correlations for %d words . . .' % (len(idx2token))
    for ii in range(len(idx2token)):

        if ii % (len(idx2token) / 4) == 0:
            print '%d/%d' % (ii, len(idx2token))

        if mask_value is not None and ii == mask_value:
            continue
        if stop_words is not None and idx2token[ii] in stop_words:
            continue

        _col = counts.getcol(ii)
        if _col.sum() == 0.0:
            zero_cnt_words += 1
            continue

        corrcoef[ii] = np.corrcoef(np.squeeze(_col.todense()), targets)[0, 1]
        mx_corrcoef = max(mx_corrcoef, corrcoef[ii])
        mn_corrcoef = min(mn_corrcoef, corrcoef[ii])

    print('Maximum correlation:', mx_corrcoef)
    print('Minimum correlation:', mn_corrcoef)
    print('Number of words with zero count:', zero_cnt_words)
    corrcoef = np.abs(corrcoef)
    corrcoef += 1e-9
    # normalize correlations to convert into distribution
    corrcoef = corrcoef / np.sum(corrcoef)
    return corrcoef


def get_mutual_information(inputs, targets, token2idx, stop_words = None, mask_token = None):

    # convert X to CSC format
    data, row, col = convert_X_to_ijv_format(inputs)
    counts = csc_matrix((data, (row, col)), shape = (inputs.shape[0], len(token2idx)))

#     tf_idf_transformer = TfidfTransformer(norm = 'l2', use_idf = True, smooth_idf = True, sublinear_tf = True)
#     tf_idf_transformer.fit(counts)
#     counts = tf_idf_transformer.transform(counts)

    mi = mutual_info_classif(counts, targets)

    mi[token2idx[mask_token]] = 0.0
    for stop_word in stop_words:
        if stop_word in token2idx:
            mi[token2idx[stop_word]] = 0.0

    print('Maximum mutual information:', np.max(mi))
    print('Minimum mutual information:', np.min(mi))
    mi += 1e-9
    return mi


def get_frequencies(counts, token2idx, stop_words = None, mask_value = None):
    assert len(counts) == len(token2idx), 'len(counts) != len(token2idx)'
    freqs = np.ones(len(counts))
    for k, v in counts.iteritems():
        idx = token2idx[k]
        if stop_words is not None and k in stop_words:
            continue
        if mask_value is not None and idx == mask_value:
            continue
        freqs[idx] += v

    freqs = freqs * 0.75
    # normalize freqs
    freqs = freqs / np.sum(freqs)
    return freqs


def get_samples(data, correlations, frequencies, num_pos = 2, num_neg = 10):
    dsz = data.shape[0]
    data = torch.from_numpy(data)
    corrs = correlations.index_select(0, data.view(-1)).view(data.size())
    sample_idx = torch.multinomial(corrs, num_pos, replacement = False)
    samples_pos = torch.gather(data.unsqueeze(2), 1, sample_idx.unsqueeze(2))
    samples_neg = torch.multinomial(frequencies, dsz * num_pos * num_neg,
                                    replacement = True).view(dsz, num_pos, num_neg)
    samples = torch.cat([samples_pos, samples_neg], 2)
    targets = torch.zeros(samples.size())
    targets[:, :, 0] = 1
    noise_probs = correlations.index_select(0, samples.view(-1)).view(samples.size())
    return (samples.numpy(), targets.numpy(), noise_probs.numpy())

# def prepare_batch(data, correlations, frequencies, num_pos = 2, num_neg = 10, batch_size = 256):
#     # data, correlations, frequencies, num_pos, num_neg, batch_size
#     samples_pos, samples_neg = get_samples(data, correlations, frequencies, num_pos = num_pos, num_neg = num_neg, batch_size = batch_size)


def get_data(corpus, min_len = 5):

    X_tr = []
    X_tr_l = []
    for X_c in corpus.unld_tr_data.X:
        if len(X_c) < min_len:
            continue
        X_tr_l.append(len(X_c))
        X_c = [corpus.pad_token_idx for _ in xrange(corpus.max_len - len(X_c))] + X_c
        X_tr.append(X_c)

    X_tr = np.asarray(X_tr)
    X_tr_l = np.asarray(X_tr_l)

    X_val = []
    X_val_l = []
    for X_c in corpus.unld_val_data.X:
        if len(X_c) < min_len:
            continue
        X_val_l.append(len(X_c))
        X_c = [corpus.pad_token_idx for _ in xrange(corpus.max_len - len(X_c))] + X_c
        X_val.append(X_c)

    X_val = np.asarray(X_val)
    X_val_l = np.asarray(X_val_l)

    print 'NCE LM training data stats:'
    print 'X_tr.shape:', X_tr.shape
    print 'X_val.shape:', X_val.shape
    return X_tr, X_tr_l, X_val, X_val_l


def batch_iter(X, X_l, correlations, frequencies, num_pos = 2, num_neg = 10, batch_size = 256, shuffle = False):
    """
    Generates a batch iterator for a dataset.
    X is padded at this point. Though the pad token is unlikely to be sampled because its frequency/correlation is set to zero.
    """
    X_size = len(X)
    frequencies = torch.from_numpy(frequencies)
    correlations = torch.from_numpy(correlations)
    num_batches_per_epoch = int((X_size - 1) / batch_size) + 1
    print 'Number of batches: ', num_batches_per_epoch
    # Shuffle the data at each epoch
    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(X_size))
        shuffled_X = X[shuffle_indices]
    else:
        shuffled_X = X

    for batch_num in range(num_batches_per_epoch):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, X_size)
        samples, targets, noise_probs = get_samples(shuffled_X[start_index:end_index], correlations, frequencies, num_pos = num_pos, num_neg = num_neg)
#         print 'samples.shape:', samples.shape
#         print 'targets.shape:', targets.shape
#         print 'noise_probs.shape:', noise_probs.shape
        yield (shuffled_X[start_index:end_index], X_l[start_index:end_index], samples, targets, noise_probs)


def binarize_labels(labels, pos_classes, label2idx):
    print 'Positive classes: ', pos_classes
    pos_classes = [label2idx[_class] for _class in pos_classes]
    labels = [1 if _label in pos_classes else 0 for _label in labels]
    return np.asarray(labels)


def train_lm(sess, model, args, corpus):

    # Define Training procedure
    global_step = tf.Variable(0, name = "global_step", trainable = False)
    # optimizer = tf.train.AdamOptimizer(1e-3)
    optimizer = NadamOptimizer()
    grads_and_vars = optimizer.compute_gradients(model.lm_loss)
    train_op = optimizer.apply_gradients(grads_and_vars, global_step = global_step)

    # Output directory for models and summaries
    if os.path.isdir(os.path.join(args['model_save_dir'], "lm_runs")):
        shutil.rmtree(os.path.join(args['model_save_dir'], "lm_runs"))
    out_dir = os.path.abspath(os.path.join(args['model_save_dir'], "lm_runs"))
    print("Writing to {}\n".format(out_dir))

    # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
    checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
    checkpoint_prefix = os.path.join(checkpoint_dir, "model")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES), max_to_keep = 1)

    # Initialize all variables
    sess.run(tf.global_variables_initializer())

    def train_step(batch_data):
        """
        A single training step based on the mode
        """
        X, _, samples, targets, noise_probs = batch_data
        feed_dict = {
          model.input_x: X,
          model.input_samples: samples,
          model.input_noise_probs: noise_probs,
          model.input_y_lm: targets,
          model.dropout_keep_prob: args['dropout']
        }
        _, step, loss = sess.run(
            [train_op, global_step, model.lm_loss],
            feed_dict)

        return loss

    def dev_step(batch_data):

        """
        Evaluates model on a dev set based on the mode
        """
        X, _, samples, targets, noise_probs = batch_data
        feed_dict = {
          model.input_x: X,
          model.input_samples: samples,
          model.input_noise_probs: noise_probs,
          model.input_y_lm: targets,
          model.dropout_keep_prob: 1.0
        }
        loss = sess.run(
            [model.lm_loss],
            feed_dict)
        return loss[0]

    X_unld_tr, X_unld_tr_l, X_unld_val, X_unld_val_l = get_data(corpus)

    stop_words = get_stopwords(args['stop_words_file'])

    targets = binarize_labels(corpus.tr_data.y, args['pos_classes'], corpus.label2idx)

    correlations = get_correlations(corpus.tr_data.X, targets, corpus.idx2token, stop_words, mask_value = corpus.pad_token_idx)
    freqs = get_frequencies(corpus.counts, corpus.token2idx, stop_words, mask_value = corpus.pad_token_idx)
    # Generate batches

    best_val_loss = 1e6
    patience = 3

    for epoch in xrange(args['n_epochs']):
        print 'Epoch %d:' % (epoch)
        _start = time.time()
        # X, correlations, frequencies, num_pos = 2, num_neg = 10, batch_size = 256, shuffle = False
        batches = batch_iter(X_unld_tr, X_unld_tr_l, correlations, freqs, batch_size = args['batch_size'], shuffle = True)
        _tr_loss = 0
        # Training loop. For each batch...
        for batch in batches:
            _tr_loss += train_step(batch)

        current_step = tf.train.global_step(sess, global_step)
        tr_loss = _tr_loss
        print 'Run time: %d s' % (time.time() - _start)
        print 'Training Loss: %f' % (tr_loss)

        val_batches = batch_iter(X_unld_val, X_unld_val_l, correlations, freqs, batch_size = args['batch_size'])
        _val_loss = 0
        for batch in val_batches:
            _val_loss += dev_step(batch)

        val_loss = _val_loss
        print 'Val Loss: %f' % (val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            path = saver.save(sess, checkpoint_prefix, global_step = current_step)
            print("Saved model checkpoint to {}\n".format(path))
            patience = 3
        else:
            patience -= 1
            print("\n")

        if patience == 0:
            print 'Early stopping . . .'
            break


def main(args):

    # train_file = None, val_file = None, test_file = None, unld_train_file = None, unld_val_file = None, dictionaries_file = None
    corpus = TweetCorpus(args['train_file'], None, None, args['unld_train_file'], args['unld_val_file'], args['dictionaries_file'])

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
                               class_weights = class_weights,
                               num_pos = args['num_pos'],
                               num_neg = args['num_neg'])

            print 'List of trainable variables:'
            for i in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
                print i.name

            train_lm(sess, cnn, args, corpus)


def parse_arguments():

    parser = argparse.ArgumentParser(description = '')
    # even though short flags can be used in the command line, they can not be used to access the value of the arguments
    # i.e args['pt'] will give KeyError.
    parser.add_argument('-tr', '--train_file', type = str, help = 'labeled train file')
    parser.add_argument('-dict', '--dictionaries_file', type = str, help = 'pickled dictionary file')
    parser.add_argument('-swf', '--stop_words_file', type = str, default = None)
    parser.add_argument('-sdir', '--model_save_dir', type = str, help = 'directory where trained model should be saved')
    parser.add_argument('-w', '--pretrained_weights', type = str, default = None, help = 'Path to pretrained weights file')
    parser.add_argument('-unld_tr', '--unld_train_file', type = str, default = None)
    parser.add_argument('-unld_val', '--unld_val_file', type = str, default = None)
    parser.add_argument('-epochs', '--n_epochs', type = int, default = 30)
    parser.add_argument('-nfmaps', '--nfeature_maps', type = int, default = 200)
    parser.add_argument('-do', '--dropout', type = float, default = 0.5)
    parser.add_argument('-bsz', '--batch_size', type = int, default = 256)
    parser.add_argument('-np', '--num_pos', type = int, default = 2, help = 'Number of positive tokens to sample per input')
    parser.add_argument('-nn', '--num_neg', type = int, default = 10, help = 'Number of negative tokens to sample per a positive token')
    parser.add_argument('-pc', '--pos-classes', nargs = '+', default = [])

    args = vars(parser.parse_args())

    return args


if __name__ == '__main__':

    main(parse_arguments())

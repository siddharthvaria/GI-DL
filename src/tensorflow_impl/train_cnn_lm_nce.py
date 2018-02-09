import argparse
import datetime
import os
import shutil
from sklearn.feature_extraction.text import TfidfTransformer
import time
import torch

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

def get_correlations(inputs, targets, idx2token, stop_words = None, mask_value = None):

    print 'Before dropping, len(inputs): ', len(inputs)
    inputs, targets = drop_tweets_by_length(inputs, targets)
    print 'After dropping, len(inputs): ', len(inputs)

    counts = np.zeros((len(idx2token), inputs.shape[0]))
    for ii in range(inputs.shape[0]):
        for jj in range(len(inputs[ii])):
            if mask_value is not None and inputs[ii][jj] == mask_value:
                continue
            counts[inputs[ii][jj]][ii] += 1

    tf_idf_transformer = TfidfTransformer(norm = 'l2', use_idf = True, smooth_idf = True, sublinear_tf = True)
    counts = np.transpose(tf_idf_transformer.fit_transform(np.transpose(counts)).toarray())
    print('counts.shape: ', counts.shape)

    zero_cnt_words = 0
    mx_corrcoef = 0.0
    mn_corrcoef = 0.0

    corrcoef = np.zeros(len(idx2token))
    for ii in range(len(idx2token)):
        if mask_value is not None and ii == mask_value:
            continue
        if stop_words is not None and idx2token[ii] in stop_words:
            continue
        if np.sum(counts[ii]) == 0.0:
            zero_cnt_words += 1
            continue
        corrcoef[ii] = np.corrcoef(counts[ii], targets)[0, 1]
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

def train_lm(sess, model, args, corpus):

    # Define Training procedure
    global_step = tf.Variable(0, name = "global_step", trainable = False)
    optimizer = tf.train.AdamOptimizer(1e-3)
    grads_and_vars = optimizer.compute_gradients(model.lm_loss)
    train_op = optimizer.apply_gradients(grads_and_vars, global_step = global_step)

    # Keep track of gradient values and sparsity (optional)
    grad_summaries = []
    for g, v in grads_and_vars:
        if g is not None:
            grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
            sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
            grad_summaries.append(grad_hist_summary)
            grad_summaries.append(sparsity_summary)
    grad_summaries_merged = tf.summary.merge(grad_summaries)

    # Output directory for models and summaries
    if os.path.isdir(os.path.join(args['model_save_dir'], "lm_runs")):
        shutil.rmtree(os.path.join(args['model_save_dir'], "lm_runs"))
    out_dir = os.path.abspath(os.path.join(args['model_save_dir'], "lm_runs"))
    print("Writing to {}\n".format(out_dir))

    # Summaries for loss and accuracy
    loss_summary = tf.summary.scalar("loss", model.lm_loss)

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
        _, step, summaries, loss = sess.run(
            [train_op, global_step, train_summary_op, model.lm_loss],
            feed_dict)
        train_summary_writer.add_summary(summaries, step)

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

    targets = np.squeeze(corpus.tr_data.y)
    np.place(targets, targets == 2, [1])
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
                               num_pos = 2,
                               num_neg = 10)

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

    args = vars(parser.parse_args())

    return args

if __name__ == '__main__':

    main(parse_arguments())

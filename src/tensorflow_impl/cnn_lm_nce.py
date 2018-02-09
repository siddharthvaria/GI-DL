import tensorflow as tf

class CNN_Model(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(
            self,
            sequence_length,
            num_classes,
            vocab_size,
            embedding_size,
            filter_sizes,
            num_filters,
            embeddings,
            class_weights,
            num_pos = 2,
            num_neg = 10):

        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name = "input_x")
        self.input_samples = tf.placeholder(tf.int32, [None, num_pos, num_neg + 1], name = "input_samples")
        self.input_noise_probs = tf.placeholder(tf.float32, [None, num_pos, num_neg + 1], name = "input_noise_probs")
        # self.input_lengths = tf.placeholder(tf.int32, [None, 1], name = "input_lengths")
        self.input_y_lm = tf.placeholder(tf.float32, [None, num_pos, num_neg + 1], name = "input_y_lm")
        self.input_y_clf = tf.placeholder(tf.float32, [None, num_classes], name = "input_y_clf")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name = "dropout_keep_prob")

        with tf.variable_scope("embedding"):
            self.W = tf.get_variable(name = "W",
                                     shape = embeddings.shape,
                                     initializer = tf.constant_initializer(embeddings),
                                     trainable = True)
            self.embedded_seqs = tf.nn.embedding_lookup(self.W, self.input_x)
            self.embedded_seqs = tf.nn.dropout(self.embedded_seqs, self.dropout_keep_prob)
            self.embedded_seqs_expanded = tf.expand_dims(self.embedded_seqs, -1)

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.variable_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev = 0.1), name = "W")
                b = tf.Variable(tf.constant(0.1, shape = [num_filters]), name = "b")
                conv = tf.nn.conv2d(
                    self.embedded_seqs_expanded,
                    W,
                    strides = [1, 1, 1, 1],
                    padding = "VALID",
                    name = "conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name = "relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize = [1, sequence_length - filter_size + 1, 1, 1],
                    strides = [1, 1, 1, 1],
                    padding = 'VALID',
                    name = "pool")
                pooled = tf.nn.dropout(pooled, self.dropout_keep_prob)
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        with tf.variable_scope("output"):
            self.h_out = tf.layers.dense(inputs = self.h_pool_flat,
                                         units = embedding_size,
                                         activation = tf.nn.relu,
                                         name = 'dense',
                                         kernel_initializer = tf.contrib.layers.xavier_initializer(),
                                         bias_initializer = tf.constant_initializer(0.1))

        # Add dropout
#         with tf.variable_scope("dropout"):
#             self.h_out = tf.nn.dropout(self.h_out, self.dropout_keep_prob)

        # Calculate nce_loss for the language model
        with tf.variable_scope("lm_loss"):
            # b_nce = tf.Variable(tf.constant(0.1, shape = [vocab_size]), name = "b_nce")

#             nce_loss = tf.nn.nce_loss(
#                 weights = W_nce,
#                 biases = b_nce,
#                 labels = self.input_y_lm,
#                 inputs = self.h_out,
#                 num_sampled = num_sampled,
#                 num_classes = vocab_size,
#                 num_true = sequence_length,
#                 partition_strategy = 'div',
#                 name = 'nce_loss'
#                 )
            W_nce = tf.get_variable(
                "W_nce",
                shape = [vocab_size, embedding_size],
                initializer = tf.random_uniform_initializer(minval = -0.25, maxval = 0.25))

            embedded_samples = tf.reshape(tf.nn.embedding_lookup(W_nce, self.input_samples), [-1, num_pos * (num_neg + 1), embedding_size])
            # At this point, embedded_samples has size  : batch_size, num_pos * (num_neg + 1), embedding_size
            score_p = tf.matmul(embedded_samples, tf.expand_dims(self.h_out, -1), name = 'score_p')
            score_p = tf.reshape(score_p, [-1, num_pos, num_neg + 1])
            # score_p has size: batch_size, num_pos, (num_neg + 1)
            score_n = tf.add(tf.log(self.input_noise_probs, name = 'score_n'), tf.log(tf.constant(num_neg * 1.0)))
            # score_n has size: batch_size, num_pos, num_neg + 1
            # TODO: figure out if masking is required here.
            logits = tf.subtract(score_p, score_n, name = 'logits')
            batch_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels = self.input_y_lm, logits = logits)
            self.lm_loss = tf.reduce_mean(batch_loss)

        # Final (unnormalized) scores and predictions
        with tf.variable_scope("clf_loss"):
            W = tf.get_variable(
                "W_cce",
                shape = [embedding_size, num_classes],
                initializer = tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape = [num_classes]), name = "b_cce")
            self.scores = tf.nn.xw_plus_b(self.h_out, W, b, name = "scores")
            self.probabilities = tf.nn.softmax(self.scores, 1, name = "probabilities")

            # use tf.losses.sparse_softmax_cross_entropy
            weights = tf.reduce_sum(tf.constant(class_weights) * self.input_y_clf, axis = 1)
            cce_loss = tf.nn.softmax_cross_entropy_with_logits(logits = self.scores, labels = self.input_y_clf)
            weighted_cce_loss = cce_loss * weights
            self.clf_loss = tf.reduce_sum(weighted_cce_loss)

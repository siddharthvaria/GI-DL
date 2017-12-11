import tensorflow as tf

class CNN_Model(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(
      self, sequence_length, num_classes, vocab_size,
      embedding_size, filter_sizes, num_filters, embeddings, num_sampled = 10):

        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name = "input_x")
        self.input_y_lm = tf.placeholder(tf.float32, [None, sequence_length], name = "input_y_lm")
        self.input_y_clf = tf.placeholder(tf.float32, [None, num_classes], name = "input_y_clf")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name = "dropout_keep_prob")
        # self.is_train = tf.placeholder(tf.int32, name = "is_train")

        # Embedding layer
        # with tf.device('/cpu:0'), tf.variable_scope("embedding"):
        with tf.variable_scope("embedding"):
            self.W = tf.get_variable(name = "W",
                                     shape = embeddings.shape,
                                     initializer = tf.constant_initializer(embeddings),
                                     trainable = True)
#             self.W = tf.Variable(
#                 tf.random_uniform([vocab_size, embedding_size], -0.25, 0.25),
#                 name = "W")
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
            # self.embedded_chars = tf.nn.dropout(self.embedded_chars, self.dropout_keep_prob)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.variable_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev = 0.1), name = "W")
                b = tf.Variable(tf.constant(0.1, shape = [num_filters]), name = "b")
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
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
                # pooled = tf.nn.dropout(pooled, self.dropout_keep_prob)
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
        with tf.variable_scope("dropout"):
            self.h_out = tf.nn.dropout(self.h_out, self.dropout_keep_prob)

        # Calculate nce_loss for the language model
        with tf.variable_scope("lm_loss"):
            W_nce = tf.get_variable(
                "W_nce",
                shape = [vocab_size, embedding_size],
                initializer = tf.contrib.layers.xavier_initializer())
            b_nce = tf.Variable(tf.constant(0.1, shape = [vocab_size]), name = "b_nce")
            nce_loss = tf.nn.nce_loss(
                weights = W_nce,
                biases = b_nce,
                labels = self.input_y_lm,
                inputs = self.h_out,
                num_sampled = num_sampled,
                num_classes = vocab_size,
                num_true = sequence_length,
                partition_strategy = 'div',
                name = 'nce_loss'
                )
            self.lm_loss = tf.reduce_sum(nce_loss)

        # Final (unnormalized) scores and predictions
        with tf.variable_scope("clf_loss"):
            W = tf.get_variable(
                "W_cce",
                shape = [embedding_size, num_classes],
                initializer = tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape = [num_classes]), name = "b_cce")
            self.scores = tf.nn.xw_plus_b(self.h_out, W, b, name = "scores")
            self.probabilities = tf.nn.softmax(self.scores, 1, name = "probabilities")
            cce_loss = tf.nn.softmax_cross_entropy_with_logits(logits = self.scores, labels = self.input_y_clf)
            self.clf_loss = tf.reduce_sum(cce_loss)

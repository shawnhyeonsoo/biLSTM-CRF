import random as rand
from random import random
import numpy as np
import math
import tensorflow as tf
#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()

MODEL_PATH = 'model/model.ckpt'


# Create a sequence classification instance.
def get_sequence(sequence_length):
    # Create a sequence of random numbers in [0,1].
    X = np.array([random() for _ in range(sequence_length)])
    # Calculate cut-off value to change class values.
    limit = sequence_length / 4.0
    # Determine the class outcome for each item in cumulative sequence.
    y = np.array([0])
    #y = np.array([0 if x < limit else 1 for x in np.cumsum(X)])

    return X, y


# Create n examples with random sequence lengths between 5 and 15.
def get_examples(n):
    X_list = []
    y_list = []
    sequence_length_list = []
    for _ in range(n):
        sequence_length = rand.randrange(start=5, stop=15)
        X, y = get_sequence(sequence_length)
        X_list.append(X)
        y_list.append(y)
        sequence_length_list.append(sequence_length)

    return X_list, y_list, sequence_length_list


# Tensorflow requires that all sentences (and all labels) inside the same batch have the same length,
# so we have to pad the data (and labels) inside the batches (with 0's, for example).
def pad(sentence, max_length):
    pad_len = max_length - len(sentence)
    padding = np.zeros(pad_len)
    return np.concatenate((sentence, padding))


# Create input batches.
def batch(data, labels, sequence_lengths, batch_size, input_size):
    n_batch = int(math.ceil(len(data) / batch_size))
    index = 0
    for _ in range(n_batch):
        batch_sequence_lengths = np.array(sequence_lengths[index: index + batch_size])
        batch_length = np.array(max(batch_sequence_lengths))  # max length in batch
        batch_data = np.array([pad(x, batch_length) for x in data[index: index + batch_size]])  # pad data
        batch_labels = np.array([pad(x, batch_length) for x in labels[index: index + batch_size]])  # pad labels
        index += batch_size

        # Reshape input data to be suitable for LSTMs.
        batch_data = batch_data.reshape(-1, batch_length, input_size)

        yield batch_data, batch_labels, batch_length, batch_sequence_lengths


x_train, y_train, sequence_length_train = get_examples(100)
x_test, y_test, sequence_length_test = get_examples(30)


# Bidirectional LSTM + CRF model.
learning_rate = 0.001
training_epochs = 1
input_size = 1
batch_size = 64
num_units = 128 # the number of units in the LSTM cell
number_of_classes = 2

input_data = tf.placeholder(tf.float32, [None, None, input_size], name="input_data") # shape = (batch, batch_seq_len, input_size)
labels = tf.placeholder(tf.int32, shape=[None, None], name="labels") # shape = (batch, sentence)
batch_sequence_length = tf.placeholder(tf.int32) # max sequence length in batch
original_sequence_lengths = tf.placeholder(tf.int32, [None])

# Scope is mandatory to use LSTMCell (https://github.com/tensorflow/tensorflow/issues/799).
with tf.name_scope("BiLSTM"):
    with tf.variable_scope('forward'):
        lstm_fw_cell = tf.nn.rnn_cell.LSTMCell(num_units, forget_bias=1.0, state_is_tuple=True)
    with tf.variable_scope('backward'):
        lstm_bw_cell = tf.nn.rnn_cell.LSTMCell(num_units, forget_bias=1.0, state_is_tuple=True)
    (output_fw, output_bw), states = tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_fw_cell,
                                                                     cell_bw=lstm_bw_cell,
                                                                     inputs=input_data,
                                                                     sequence_length=original_sequence_lengths,
                                                                     dtype=tf.float32,
                                                                     scope="BiLSTM")

# As we have a Bi-LSTM, we have two outputs which are not connected, so we need to merge them.
outputs = tf.concat([output_fw, output_bw], axis=2)

# Fully connected layer.
W = tf.get_variable(name="W", shape=[2 * num_units, number_of_classes],
                dtype=tf.float32)

b = tf.get_variable(name="b", shape=[number_of_classes], dtype=tf.float32,
                initializer=tf.zeros_initializer())

outputs_flat = tf.reshape(outputs, [-1, 2 * num_units])
pred = tf.matmul(outputs_flat, W) + b
scores = tf.reshape(pred, [-1, batch_sequence_length, number_of_classes])

# Linear-CRF.
log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(scores, labels, original_sequence_lengths)

loss = tf.reduce_mean(-log_likelihood)

# Compute the viterbi sequence and score (used for prediction and test time).
viterbi_sequence, viterbi_score = tf.contrib.crf.crf_decode(scores, transition_params, original_sequence_lengths)

# Training ops.
optimizer = tf.train.AdamOptimizer(learning_rate)
train_op = optimizer.minimize(loss)

# Add ops to save and restore all the variables.
saver = tf.train.Saver()

# Training the model.
with tf.Session() as session:
    session.run(tf.global_variables_initializer())

    for i in range(training_epochs):
        for batch_data, batch_labels, batch_seq_len, batch_sequence_lengths in batch(x_train, y_train,
                                                                                     sequence_length_train, batch_size,
                                                                                     input_size):
            tf_viterbi_sequence, _ = session.run([viterbi_sequence, train_op],
                                                 feed_dict={input_data: batch_data,
                                                            labels: batch_labels,
                                                            batch_sequence_length: batch_seq_len,
                                                            original_sequence_lengths: batch_sequence_lengths})
            # Show train accuracy.
            if i % 10 == 0:
                # Create a mask to fix input lengths.
                mask = (np.expand_dims(np.arange(batch_seq_len), axis=0) <
                        np.expand_dims(batch_sequence_lengths, axis=1))
                total_labels = np.sum(batch_sequence_lengths)
                correct_labels = np.sum((batch_labels == tf_viterbi_sequence) * mask)
                accuracy = 100.0 * correct_labels / float(total_labels)
                print("Epoch: %d" % i, "Accuracy: %.2f%%" % accuracy)

    # Save the variables to disk.
    saver.save(session, MODEL_PATH)

# Testing the model.
with tf.Session() as session:
    # Restore variables from disk.
    saver.restore(session, MODEL_PATH)

    for batch_data, batch_labels, batch_seq_len, batch_sequence_lengths in batch(x_test, y_test, sequence_length_test,
                                                                                 len(x_test), input_size):
        tf_viterbi_sequence = session.run(viterbi_sequence, feed_dict={input_data: batch_data,
                                                                       labels: batch_labels,
                                                                       batch_sequence_length: batch_seq_len,
                                                                       original_sequence_lengths: batch_sequence_lengths})
    # mask to correct input sizes
    mask = (np.expand_dims(np.arange(batch_seq_len), axis=0) <
            np.expand_dims(batch_sequence_lengths, axis=1))
    total_labels = np.sum(batch_sequence_lengths)
    correct_labels = np.sum((batch_labels == tf_viterbi_sequence) * mask)
    accuracy = 100.0 * correct_labels / float(total_labels)
    print("Test accuracy: %.2f%%" % accuracy)

    print("Label:", batch_labels[0].astype(int))
    print("Pred.:", tf_viterbi_sequence[0])

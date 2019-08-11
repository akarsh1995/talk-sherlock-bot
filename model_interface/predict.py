import numpy as np
import pickle
import tensorflow as tf
from model_interface.model_interface import get_test_input, ids_to_sentence
import config

# Load in data structures
with open(config.word_list_filepath, "rb") as fp:
    word_list = pickle.load(fp)
word_list.append('<pad>')
word_list.append('<EOS>')

# Load in hyperparamters

vocab_size = len(word_list)
batchSize = 24
max_encoder_length = 15
max_decoder_length = 15
lstm_units = 112
num_layer_lstm = 3

# Create placeholders
encoder_inputs = [tf.placeholder(tf.int32, shape=(None,)) for i in range(max_encoder_length)]
decoder_labels = [tf.placeholder(tf.int32, shape=(None,)) for i in range(max_decoder_length)]
decoder_inputs = [tf.placeholder(tf.int32, shape=(None,)) for i in range(max_decoder_length)]
feed_previous = tf.placeholder(tf.bool)

encoder_lstm = tf.nn.rnn_cell.BasicLSTMCell(lstm_units, state_is_tuple=True)
# encoder_lstm = tf.nn.rnn_cell.MultiRNNCell([singleCell]*num_layer_lstm, state_is_tuple=True)
decoderOutputs, decoderFinalState = tf.contrib.legacy_seq2seq.embedding_rnn_seq2seq(encoder_inputs, decoder_inputs,
                                                                                    encoder_lstm,
                                                                                    vocab_size, vocab_size, lstm_units,
                                                                                    feed_previous=feed_previous)

decoderPrediction = tf.argmax(decoderOutputs, 2)

# Start session and get graph
sess = tf.Session()
# y, variables = model_interface.getmodel_interface(encoder_inputs, decoder_labels, decoder_inputs, feed_previous)

# Load in pretrained model
saver = tf.train.Saver()
saver.restore(sess, tf.train.latest_checkpoint(config.models_dir))
zero_vector = np.zeros(1, dtype='int32')


def pred(input_string):
    inputVector = get_test_input(input_string, word_list, max_encoder_length)
    feed_dict = {encoder_inputs[t]: inputVector[t] for t in range(max_encoder_length)}
    feed_dict.update({decoder_labels[t]: zero_vector for t in range(max_decoder_length)})
    feed_dict.update({decoder_inputs[t]: zero_vector for t in range(max_decoder_length)})
    feed_dict.update({feed_previous: True})
    ids = (sess.run(decoderPrediction, feed_dict=feed_dict))
    return ids_to_sentence(ids, word_list)

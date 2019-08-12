import numpy as np
import pickle
import os
import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)  # or any {DEBUG, INFO, WARN, ERROR, FATAL}
from model_interface.model_interface import get_test_input, ids_to_sentence
import config

# Load in data structures
with open(config.word_list_filepath, "rb") as fp:
    word_list = pickle.load(fp)
word_list.append('<pad>')
word_list.append('<EOS>')

# Load in hyperparamters

hp = config.HyperParameters()
hp.vocab_size = hp.vocab_size() + 2
# Create placeholders
encoder_inputs = [tf.placeholder(tf.int32, shape=(None,)) for i in range(hp.max_encoder_length)]
decoder_labels = [tf.placeholder(tf.int32, shape=(None,)) for i in range(hp.max_decoder_length)]
decoder_inputs = [tf.placeholder(tf.int32, shape=(None,)) for i in range(hp.max_decoder_length)]
feed_previous = tf.placeholder(tf.bool)

encoder_lstm = tf.nn.rnn_cell.BasicLSTMCell(hp.lstm_units, state_is_tuple=True)
# encoder_lstm = tf.nn.rnn_cell.MultiRNNCell([singleCell]*num_layer_lstm, state_is_tuple=True)
decoder_outputs, decoder_final_state = tf.contrib.legacy_seq2seq.embedding_rnn_seq2seq(encoder_inputs, decoder_inputs,
                                                                                       encoder_lstm,
                                                                                       hp.vocab_size, hp.vocab_size,
                                                                                       hp.lstm_units,
                                                                                       feed_previous=feed_previous)
decoder_prediction = tf.argmax(decoder_outputs, 2)

# Start session and get graph
# y, variables = model_interface.getmodel_interface(encoder_inputs, decoder_labels, decoder_inputs, feed_previous)
zero_vector = np.zeros(1, dtype='int32')
sess = tf.Session()
if os.listdir(config.models_dir):
    saver = tf.train.Saver()
    saver.restore(sess, tf.train.latest_checkpoint(config.models_dir))


def get_feed_dict(inputVector, max_encoder_length, max_decoder_length):
    feed_dict = {encoder_inputs[t]: inputVector[t] for t in range(max_encoder_length)}
    feed_dict.update({decoder_labels[t]: zero_vector for t in range(max_decoder_length)})
    feed_dict.update({decoder_inputs[t]: zero_vector for t in range(max_decoder_length)})
    feed_dict.update({feed_previous: True})
    return feed_dict


def pred(input_string, decoder_pred=decoder_prediction):
    inputVector = get_test_input(input_string, word_list, hp.max_encoder_length)
    feed_dict = get_feed_dict(inputVector, hp.max_encoder_length, hp.max_decoder_length)
    ids = (sess.run(decoder_pred, feed_dict=feed_dict))
    return ids_to_sentence(ids, word_list)


def predict_output(input_string):
    return pred(input_string, decoder_prediction)

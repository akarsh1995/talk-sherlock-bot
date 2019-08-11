import tensorflow as tf
import numpy as np
from random import randint
import datetime
import pickle
import os
import config

from model_interface.predict import pred as predict

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def create_training_matrices(dialogue_conversation_dict_file, w_list, max_len):
    conversation_dictionary = np.load(dialogue_conversation_dict_file, allow_pickle=True).item()
    numExamples = len(conversation_dictionary)
    x_train_ = np.zeros((numExamples, max_len), dtype='int32')
    y_train_ = np.zeros((numExamples, max_len), dtype='int32')
    for index, (key, value) in enumerate(conversation_dictionary.items()):
        # Will store integerized representation of strings here (initialized as padding)
        encoder_message = np.full(max_len, w_list.index('<pad>'), dtype='int32')
        decoder_message = np.full(max_len, w_list.index('<pad>'), dtype='int32')
        # Getting all the individual words in the strings
        key_split = key.split()
        value_split = value.split()
        key_count = len(key_split)
        value_count = len(value_split)
        # Throw out sequences that are too long or are empty
        if key_count > (max_len - 1) or value_count > (max_len - 1) or value_count == 0 or key_count == 0:
            continue
        # Integerize the encoder string
        for key_index, word in enumerate(key_split):
            try:
                encoder_message[key_index] = w_list.index(word)
            except ValueError:
                encoder_message[key_index] = 0
        encoder_message[key_index + 1] = w_list.index('<EOS>')
        # Integerize the decoder string
        for value_index, word in enumerate(value_split):
            try:
                decoder_message[value_index] = w_list.index(word)
            except ValueError:
                decoder_message[value_index] = 0
        decoder_message[value_index + 1] = w_list.index('<EOS>')
        x_train_[index] = encoder_message
        y_train_[index] = decoder_message
    # Remove rows with all zeros
    y_train_ = y_train_[~np.all(y_train_ == 0, axis=1)]
    x_train_ = x_train_[~np.all(x_train_ == 0, axis=1)]
    numExamples = x_train_.shape[0]
    return numExamples, x_train_, y_train_


def get_training_batch(local_x_train, local_y_train, local_batch_size, max_len):
    num_ = randint(0, num_training_examples - local_batch_size - 1)
    arr = local_x_train[num_:num_ + local_batch_size]
    labels = local_y_train[num_:num_ + local_batch_size]
    # Reversing the order of encoder string apparently helps as per 2014 paper
    reversed_list = list(arr)
    for index, example in enumerate(reversed_list):
        reversed_list[index] = list(reversed(example))

    # Lagged labels are for the training input into the decoder
    lagged_labels = []
    eos_token_index = word_list.index('<EOS>')
    pad_token_index = word_list.index('<pad>')
    for example in labels:
        eos_found = np.argwhere(example == eos_token_index)[0]
        shifted_example = np.roll(example, 1)
        shifted_example[0] = eos_token_index
        # The EOS token was already at the end, so no need for pad
        if eos_found != (max_len - 1):
            shifted_example[eos_found + 1] = pad_token_index
        lagged_labels.append(shifted_example)

    # Need to transpose these
    reversed_list = np.asarray(reversed_list).T.tolist()
    labels = labels.T.tolist()
    lagged_labels = np.asarray(lagged_labels).T.tolist()
    return reversed_list, labels, lagged_labels


def translate_to_sentences(inputs, w_list, encoder=False):
    eos_token_index = w_list.index('<EOS>')
    pad_token_index = w_list.index('<pad>')
    numStrings = len(inputs[0])
    num_length_of_strings = len(inputs)
    list_of_strings = [''] * numStrings
    for mySet in inputs:
        for index, num_ in enumerate(mySet):
            if num_ != eos_token_index and num_ != pad_token_index:
                if encoder:
                    # Encodings are in reverse!
                    list_of_strings[index] = w_list[num_] + " " + list_of_strings[index]
                else:
                    list_of_strings[index] = list_of_strings[index] + " " + w_list[num_]
    list_of_strings = [string.strip() for string in list_of_strings]
    return list_of_strings


# Hyperparamters
batch_size = 24
max_encoder_length = 15
max_decoder_length = max_encoder_length
lstm_units = 112
embedding_dim = lstm_units
num_layers_lstm = 3
num_iterations = 500000

# Loading in all the data structures
with open(config.word_list_filepath, "rb") as fp:
    word_list = pickle.load(fp)

vocab_size = len(word_list)

# If you've run the entirety of word2vec.py then these lines will load in
# the embedding matrix.
if os.path.isfile(config.embedding_matrix_filepath):
    word_vectors = np.load(config.embedding_matrix_filepath)
    wordVecDimensions = word_vectors.shape[1]
else:
    question = 'Since we cant find an embedding matrix, how many dimensions do you want your word vectors to be?: '
    wordVecDimensions = int(input(question))

# Add two entries to the word vector matrix. One to represent padding tokens,
# and one to represent an end of sentence token
pad_vector = np.zeros((1, wordVecDimensions), dtype='int32')
eos_vector = np.ones((1, wordVecDimensions), dtype='int32')
if os.path.isfile(config.embedding_matrix_filepath):
    word_vectors = np.concatenate((word_vectors, pad_vector), axis=0)
    word_vectors = np.concatenate((word_vectors, eos_vector), axis=0)

# Need to modify the word list as well
word_list.append('<pad>')
word_list.append('<EOS>')
vocab_size = vocab_size + 2

if os.path.isfile(config.seq2seq_x_train_filepath) and os.path.isfile(config.seq2seq_y_train_filepath):
    x_train = np.load(config.seq2seq_x_train_filepath)
    y_train = np.load(config.seq2seq_y_train_filepath)
    print('Finished loading training matrices')
    num_training_examples = x_train.shape[0]
else:
    num_training_examples, x_train, y_train = create_training_matrices(config.conversation_dictionary_filepath,
                                                                       word_list,
                                                                       max_encoder_length)
    np.save(config.seq2seq_x_train_filepath, x_train)
    np.save(config.seq2seq_y_train_filepath, y_train)
    print('Finished creating training matrices')

tf.reset_default_graph()

# Create the placeholders
encoder_inputs = [tf.placeholder(tf.int32, shape=(None,)) for i in range(max_encoder_length)]
decoder_labels = [tf.placeholder(tf.int32, shape=(None,)) for i in range(max_decoder_length)]
decoder_inputs = [tf.placeholder(tf.int32, shape=(None,)) for i in range(max_decoder_length)]
feed_previous = tf.placeholder(tf.bool)

encoder_lstm = tf.nn.rnn_cell.BasicLSTMCell(lstm_units, state_is_tuple=True)

# encoder_lstm = tf.nn.rnn_cell.MultiRNNCell([singleCell]*num_layers_lstm, state_is_tuple=True)
# Architectural choice of of whether or not to include ^

decoder_outputs, decoderFinalState = tf.contrib.legacy_seq2seq.embedding_rnn_seq2seq(encoder_inputs, decoder_inputs,
                                                                                     encoder_lstm,
                                                                                     vocab_size, vocab_size,
                                                                                     embedding_dim,
                                                                                     feed_previous=feed_previous)

decoder_prediction = tf.argmax(decoder_outputs, 2)

loss_weights = [tf.ones_like(l, dtype=tf.float32) for l in decoder_labels]
loss = tf.contrib.legacy_seq2seq.sequence_loss(decoder_outputs, decoder_labels, loss_weights, vocab_size)
optimizer = tf.train.AdamOptimizer(1e-4).minimize(loss)

sess = tf.Session()
saver = tf.train.Saver()
# If you're loading in a saved model, uncomment the following line and comment out line 202
# saver.restore(sess, tf.train.latest_checkpoint('models/'))
sess.run(tf.global_variables_initializer())

# Uploading results to Tensorboard
tf.summary.scalar('Loss', loss)
merged = tf.summary.merge_all()
log_dir = os.path.join(config.tensor_board_dir, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
writer = tf.summary.FileWriter(log_dir, sess.graph)

# Some test strings that we'll use as input at intervals during training
encoder_test_strings = ["murder is a mystry",
                        "mycroft is the government"
                        "hey Sherlock",
                        "high functioning sociopath",
                        "Watson",
                        "Afghanistan"
                        ]

zero_vector = np.zeros(1, dtype='int32')

for i in range(num_iterations):

    encoder_train, decoder_target_train, decoder_input_train = get_training_batch(x_train, y_train, batch_size,
                                                                                  max_encoder_length)
    feed_dict = {encoder_inputs[t]: encoder_train[t] for t in range(max_encoder_length)}
    feed_dict.update({decoder_labels[t]: decoder_target_train[t] for t in range(max_decoder_length)})
    feed_dict.update({decoder_inputs[t]: decoder_input_train[t] for t in range(max_decoder_length)})
    feed_dict.update({feed_previous: False})

    curLoss, _, pred = sess.run([loss, optimizer, decoder_prediction], feed_dict=feed_dict)

    if i % 50 == 0:
        print('Current loss:', curLoss, 'at iteration', i)
        summary = sess.run(merged, feed_dict=feed_dict)
        writer.add_summary(summary, i)
    if i % 25 == 0 and i != 0:
        num = randint(0, len(encoder_test_strings) - 1)
        print(encoder_test_strings[num])
        input_string = encoder_test_strings[num]
        print(predict(input_string))

    if i % 10000 == 0 and i != 0:
        savePath = saver.save(sess, os.path.join(config.models_dir, "pretrained_seq2seq.ckpt"), global_step=i)

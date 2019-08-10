import tensorflow as tf
import numpy as np
from collections import Counter
import sys
import math
from random import randint
import pickle
import os
import tensorflow.python.framework.dtypes

# This Word2Vec implementation is largely based on this paper
# https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf
# It's a bit old, but Word2Vec is still SOTA and relatively simple, so I'm going with it

# Check out Tensorflow's documentation which is pretty good for Word2Vec
# https://www.tensorflow.org/tutorials/word2vec

word_vec_dimensions = 100
batch_size = 128
num_negative_sample = 64
window_size = 5
num_iterations = 100000


# This function just takes in the conversation data and makes it
# into one huge string, and then uses a Counter to identify words
# and the number of occurences
def process_dataset(filename):
    opened_file = open(filename, 'r')
    all_lines = opened_file.readlines()
    my_str = ""
    for line in all_lines:
        my_str += line
    final_dict = Counter(my_str.split())
    return my_str, final_dict


def create_training_matrices(dictionary, corpus):
    all_unique_words = list(dictionary.keys())
    all_words = corpus.split()
    num_total_words = len(all_words)
    x_train_ = []
    y_train_ = []
    for i_ in range(num_total_words):
        if i_ % 100000 == 0:
            print('Finished %d/%d total words' % (i_, num_total_words))
        words_after = all_words[i_ + 1:i_ + window_size + 1]
        words_before = all_words[max(0, i_ - window_size):i_]
        words_added = words_after + words_before
        for word in words_added:
            x_train_.append(all_unique_words.index(all_words[i_]))
            y_train_.append(all_unique_words.index(word))
    return x_train_, y_train_


def get_training_batch():
    num = randint(0, num_training_examples - batch_size - 1)
    arr = x_train[num:num + batch_size]
    labels = y_train[num:num + batch_size]
    return arr, labels[:, np.newaxis]


continue_word2vec = True
# Loading the data structures if they are present in the directory
if os.path.isfile('word2vec_x_train.npy') and os.path.isfile('word2vec_y_train.npy') and os.path.isfile('word_list.txt'):
    x_train = np.load('word2vec_x_train.npy')
    y_train = np.load('word2vec_y_train.npy')
    print('Finished loading training matrices')
    with open("word_list.txt", "rb") as fp:
        word_list = pickle.load(fp)
    print('Finished loading word list')

else:
    full_corpus, dataset_dictionary = process_dataset('dataset/Sherlock/cleaned_data.txt')
    print('Finished parsing and cleaning dataset')
    word_list = list(dataset_dictionary.keys())
    create_own_vectors = input('Do you want to create your own vectors through Word2Vec (y/n)?')
    if create_own_vectors == 'y':
        x_train, y_train = create_training_matrices(dataset_dictionary, full_corpus)
        print('Finished creating training matrices')
        np.save('word2vec_x_train.npy', x_train)
        np.save('word2vec_y_train.npy', y_train)
    else:
        continue_word2vec = False
    with open("word_list.txt", "wb") as fp:
        pickle.dump(word_list, fp)

# If you do not want to create your own word vectors and you'd just like to 
# have Tensorflow's seq2seq take care of that, then you don't need to run 
# anything below this line. 
if not continue_word2vec:
    sys.exit()

num_training_examples = len(x_train)
vocabSize = len(word_list)

sess = tf.Session()
embeddingMatrix = tf.Variable(tf.random_uniform([vocabSize, word_vec_dimensions], -1.0, 1.0))
nceWeights = tf.Variable(tf.truncated_normal([vocabSize, word_vec_dimensions], stddev=1.0 / math.sqrt(word_vec_dimensions)))
nceBiases = tf.Variable(tf.zeros([vocabSize]))

inputs = tf.placeholder(tf.int32, shape=[batch_size])
outputs = tf.placeholder(tf.int32, shape=[batch_size, 1])

embed = tf.nn.embedding_lookup(embeddingMatrix, inputs)

loss = tf.reduce_mean(
    tf.nn.nce_loss(weights=nceWeights,
                   biases=nceBiases,
                   labels=outputs,
                   inputs=embed,
                   num_sampled=num_negative_sample,
                   num_classes=vocabSize))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0).minimize(loss)

sess.run(tf.global_variables_initializer())
for i in range(num_iterations):
    trainInputs, trainLabels = get_training_batch()
    _, curLoss = sess.run([optimizer, loss], feed_dict={inputs: trainInputs, outputs: trainLabels})
    if i % 10000 == 0:
        print('Current loss is:', curLoss)
print('Saving the word embedding matrix')
embedMatrix = embeddingMatrix.eval(session=sess)
np.save('embeddingMatrix.npy', embedMatrix)

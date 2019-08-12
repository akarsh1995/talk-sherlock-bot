import os
import pickle
data_dir = 'data'
subtitles_dir = os.path.join(data_dir, 'Sherlock')
models_dir = os.path.join(data_dir, 'models')
tensor_board_dir = os.path.join(data_dir, 'tensorboard')

if not os.path.exists(models_dir):
    os.mkdir(models_dir)
if not os.path.exists(data_dir):
    os.mkdir(data_dir)

conversation_dictionary_filepath = os.path.join(data_dir, 'conversation_dictionary.npy')
embedding_matrix_filepath = os.path.join(data_dir, 'embedding_matrix.npy')
seq2seq_x_train_filepath = os.path.join(data_dir, 'seq2seq_x_train.npy')
seq2seq_y_train_filepath = os.path.join(data_dir, 'seq2seq_y_train.npy')
word2vec_x_train_filepath = os.path.join(data_dir, 'word2vec_x_train.npy')
word2vec_y_train_filepath = os.path.join(data_dir, 'word2vec_y_train.npy')
word_list_filepath = os.path.join(data_dir, 'word_list.npy')
cleaned_conversation_txt = os.path.join(data_dir, 'cleaned_data.txt')


def get_word_list_length():
    with open(word_list_filepath, "rb") as fp:
        word_list = pickle.load(fp)
    return len(word_list)


hyper_parameters = {
    "vocab_size": get_word_list_length,
    "batch_size": 24,
    "max_encoder_length": 15,
    "max_decoder_length": 15,
    "lstm_units": 112,
    "num_layer_lstm": 3,
    "embedding_dim": 112,
    "num_iterations": 500000
}


class HyperParameters:
    vocab_size: int
    batch_size: int
    max_encoder_length: int
    max_decoder_length: int
    lstm_units: int
    num_layer_lstm: int
    embedding_dim: int
    num_iterations: int

    def __init__(self):
        for k, v in hyper_parameters.items():
            setattr(self, k, v)

import os

data_dir = 'data'
subtitles_dir = './dataset/Sherlock'
models_dir = 'models'
tensor_board_dir = 'tensorboard/'

conversation_dictionary_filepath = os.path.join(data_dir, 'conversation_dictionary.npy')
embedding_matrix_filepath = os.path.join(data_dir, 'embedding_matrix.npy')
seq2seq_x_train_filepath = os.path.join(data_dir, 'seq2seq_x_train.npy')
seq2seq_y_train_filepath = os.path.join(data_dir, 'seq2seq_y_train.npy')
word2vec_x_train_filepath = os.path.join(data_dir, 'word2vec_x_train.npy')
word2vec_y_train_filepath = os.path.join(data_dir, 'word2vec_y_train.npy')
word_list_filepath = os.path.join(data_dir, 'word_list.txt')
cleaned_conversation_txt = os.path.join(data_dir, 'cleaned_data.txt')

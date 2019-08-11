import numpy as np


def get_test_input(input_message, w_list, max_len):
    encoderMessage = np.full(max_len, w_list.index('<pad>'), dtype='int32')
    inputSplit = input_message.lower().split()
    for index, word in enumerate(inputSplit):
        try:
            encoderMessage[index] = w_list.index(word)
        except ValueError:
            continue
    encoderMessage[index + 1] = w_list.index('<EOS>')
    encoderMessage = encoderMessage[::-1]
    encoderMessageList = []
    for num_ in encoderMessage:
        encoderMessageList.append([num_])
    return encoderMessageList


def ids_to_sentence(ids_, w_list):
    eos_token_index = w_list.index('<EOS>')
    pad_token_index = w_list.index('<pad>')
    my_str = ""
    list_of_responses = []
    for num_ in ids_:
        if num_[0] == eos_token_index or num_[0] == pad_token_index:
            list_of_responses.append(my_str)
            my_str = ""
        else:
            my_str = my_str + w_list[num_[0]] + " "
    if my_str:
        list_of_responses.append(my_str)
    list_of_responses = [i for i in list_of_responses if i]
    return list_of_responses

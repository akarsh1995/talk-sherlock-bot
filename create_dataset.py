import os
import re
import numpy as np
import config


def clean_subtitle(message):
    # Remove new lines within message
    cleaned_dialogue = message.replace('\n', ' ').lower()
    # Deal with some weird tokens
    cleaned_dialogue = cleaned_dialogue.replace("\xc2\xa0", "")
    # Remove punctuation
    cleaned_dialogue = re.sub('([.,!?])', '', cleaned_dialogue)
    # Remove multiple spaces in message
    cleaned_dialogue = re.sub(' +', ' ', cleaned_dialogue)
    return cleaned_dialogue


def startswith_alpha(line):
    """only lines starting with alphabet"""
    return bool(re.match(r'^[A-z]', line))


class SubCleaner:
    """Cleans the subtitles files and concatenate all files into one."""
    files_paths = []

    def __init__(self, subtitles_dir):
        self.sub_dir = subtitles_dir
        self.files_paths = self.get_files_path()
        self.cleaned_file = open(os.path.join(self.sub_dir, config.cleaned_conversation_txt), 'w')

    def get_files_path(self):
        files = os.listdir(self.sub_dir)
        files_paths = [os.path.join(self.sub_dir, file) for file in files if file.endswith('.srt')]
        return files_paths

    def clean_files(self):
        for file in self.files_paths:
            self._clean_file(file)
        print('cleaned all files successfully')
        sc.cleaned_file.close()

    def _clean_file(self, file_path):
        with open(file_path, 'r') as cf:
            line = cf.readline()
            while line:
                if startswith_alpha(line):
                    cleaned_line = clean_subtitle(line)
                    self.write_line(cleaned_line)
                line = cf.readline()
            print(f"cleaned {os.path.basename(file_path)}")

    def write_line(self, cleaned_line):
        self.cleaned_file.write(cleaned_line + '\n')


sc = SubCleaner(config.subtitles_dir)

sc.clean_files()


def create_conversation_dict(filepath):
    """Creates dictionary of the response and message present in every second line"""
    conversation_dict = {}
    with open(filepath, 'r') as f:
        sherlock = f.readline()
        other = f.readline()

        while sherlock and other:
            conversation_dict[other] = sherlock
            sherlock = f.readline()
            other = f.readline()
    return conversation_dict


np.save(config.conversation_dictionary_filepath, create_conversation_dict(config.cleaned_conversation_txt))

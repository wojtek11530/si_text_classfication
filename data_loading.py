import os
import numpy as np


def load_train_data():
    directory_path = 'data'
    subdirectory_path = 'wiki_train_34_categories_data'
    path = os.path.join(directory_path, subdirectory_path)
    texts, categories = load_data(path)
    return texts, categories


def load_test_data():
    directory_path = 'data'
    subdirectory_path = 'wiki_test_34_categories_data'
    path = os.path.join(directory_path, subdirectory_path)
    texts, categories = load_data(path)
    return texts, categories


def load_data(path):
    file_names = get_file_names_from_directory(path)
    categories = np.array([file_name.split('_')[0] for file_name in file_names])

    i_max = None
    texts = get_texts_from_files(path, file_names, i_max)

    if i_max is not None:
        categories = categories[:i_max]
    return texts, categories


def get_file_names_from_directory(directory_path):
    file_names = [file_name for file_name in os.listdir(directory_path) if is_correct_txt_file(directory_path, file_name)]
    return file_names


def get_texts_from_files(path, file_names, i_max=None):
    all_texts = []

    i = 0
    for file_name in file_names:
        directory = os.path.join(path, file_name)
        file = open(directory, encoding="utf8")
        text = file.read()
        all_texts.append(text)
        file.close()
        i += 1
        if i_max is not None and i >= i_max:
            break
    return all_texts


def is_correct_txt_file(directory_path, file_name):
    return os.path.isfile(os.path.join(directory_path, file_name)) and os.path.splitext(file_name)[1] == '.txt'\
           and os.path.splitext(file_name)[0] != 'license'

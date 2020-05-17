from data_loading import load_train_data, load_test_data
from feature_extraction import *
import classifying

import numpy as np
import matplotlib.pyplot as plt


def run():
    texts, categories = load_train_data()
    test_texts, test_categories = load_test_data()

    texts = preprocess(texts)
    vocabulary_vectorizer = get_vocabulary_vectorizer(texts)
    all_features = get_all_features(texts, vocabulary_vectorizer)
    test_texts = preprocess(test_texts)
    test_all_features = get_all_features(test_texts, vocabulary_vectorizer)

    k_best_features_nums = [500, 1000, 5000, 10000, 20000, 30000, 35000, 40000,
                            60000, 80000, 100000, 150000, 200000, all_features.shape[1]]
    accuracies = []
    for k in k_best_features_nums:
        feature_selector = get_feature_selector(all_features, categories, k)
        selected_features = get_selected_features(all_features, feature_selector)
        clf_model = classifying.get_multinomial_nb_model(selected_features, categories)
        test_selected_features = get_selected_features(test_all_features, feature_selector)
        accuracy = classifying.predict(clf_model, test_selected_features, test_categories, False)
        accuracies.append(accuracy)

    plt.plot(k_best_features_nums, accuracies, '.--')
    plt.xlabel(r'number of selected features $k$')
    plt.ylabel('accuracy')
    plt.xscale('log')
    plt.grid()
    plt.show()


if __name__ == '__main__':
    run()

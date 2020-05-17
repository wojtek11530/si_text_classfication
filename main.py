from data_loading import load_train_data, load_test_data
from feature_extraction import preprocess, get_features, get_vocabulary_vectorizer
import classifying

import os
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer


def run():
    texts, categories = load_train_data()
    texts = preprocess(texts)
    vocabulary_vectorizer = get_vocabulary_vectorizer(texts)
    features = get_features(texts, vocabulary_vectorizer)
    clf_model = classifying.get_multinomial_nb_model(features, categories)

    test_texts, test_categories = load_test_data()
    test_texts = preprocess(test_texts)
    test_features = get_features(test_texts, vocabulary_vectorizer)
    classifying.predict(clf_model, test_features, test_categories)


if __name__ == '__main__':
    run()

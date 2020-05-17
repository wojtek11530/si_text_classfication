import re

from sklearn.feature_extraction.text import CountVectorizer
import stop_words
from sklearn.feature_selection import chi2, SelectKBest


def get_vocabulary_vectorizer(texts):
    polish_stopwords = stop_words.get_stop_words('polish')
    vectorizer = CountVectorizer(min_df=1, max_df=1.0, stop_words=polish_stopwords, analyzer='word')
    vectorizer.fit(texts)
    return vectorizer


def get_all_features(texts, vectorizer):
    features = vectorizer.transform(texts)
    return features


def get_feature_selector(all_features, category_labels, k=10000):
    chi2_selector = SelectKBest(chi2, k=k)
    k_best_features_selector = chi2_selector.fit(all_features, category_labels)
    return k_best_features_selector


def get_selected_features(all_features, feature_selector):
    return feature_selector.transform(all_features)


def preprocess(texts):
    preprocessed_texts = []
    for text in texts:
        # Remove all the special characters, punctuation signs
        text = re.sub(r'\W+', ' ', text)
        # Replace not single space with single one
        text = re.sub(r'\s+', ' ', text)
        # To lowercase
        text = text.lower()

        preprocessed_texts.append(text)
    return preprocessed_texts

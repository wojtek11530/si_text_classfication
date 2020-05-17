import re

from sklearn.feature_extraction.text import CountVectorizer
import stop_words


def get_vocabulary_vectorizer(texts):

    polish_stopwords = stop_words.get_stop_words('polish')
    vectorizer = CountVectorizer(min_df=1, max_df=1.0, stop_words=polish_stopwords, analyzer='word')
    vectorizer.fit(texts)
    return vectorizer


def get_features(texts, vectorizer):
    features = vectorizer.transform(texts)
    return features


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



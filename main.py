from classifying import *
from data_loading import load_all_data
from feature_extraction import *

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn import tree


def run_for_var_feat_num():
    texts, categories = load_all_data()
    texts = preprocess(texts)
    train_texts, test_texts, train_categories, test_categories = train_test_split(texts, categories, test_size=0.2,
                                                                                  random_state=0)
    vocabulary_vectorizer = get_vocabulary_vectorizer(train_texts)
    # vocabulary_binary_vectorizer = get_vocabulary_vectorizer(train_texts, binary=True)

    train_all_features = get_all_features(train_texts, vocabulary_vectorizer)
    test_all_features = get_all_features(test_texts, vocabulary_vectorizer)

    cv_result_tuple = []
    models = [MultinomialNB(alpha=0.1), tree.DecisionTreeClassifier(max_depth=10)]

    features_nums = [
        [2000, 5000, 10000, 20000, 30000, 40000, 60000, 80000],
        [2000, 5000, 10000, 20000, 30000, 40000]]

    for model, nums in zip(models, features_nums):
        avg_cv_accuracies = perform_cv_for_various_features_num(model, train_all_features, train_categories, nums)
        cv_result_tuple.append((model, nums, avg_cv_accuracies))

    plot_cv_results(cv_result_tuple)
    for model, features_nums, avg_cv_accuracies in cv_result_tuple:
        best_features_num = determine_the_best_features_num(features_nums, avg_cv_accuracies)
        evaluate_on_test_data(model, test_all_features, test_categories, train_all_features, train_categories,
                              best_features_num)


def single_run():
    texts, categories = load_all_data()
    texts = preprocess(texts)
    train_texts, test_texts, train_categories, test_categories = train_test_split(texts, categories, test_size=0.2,
                                                                                  random_state=0)
    vocabulary_vectorizer = get_vocabulary_vectorizer(train_texts)
    vocabulary_binary_vectorizer = get_vocabulary_vectorizer(train_texts, binary=True)

    train_all_features = get_all_features(train_texts, vocabulary_vectorizer)
    test_all_features = get_all_features(test_texts, vocabulary_vectorizer)

    train_all_binary_features = get_all_features(train_texts, vocabulary_vectorizer)
    test_all_binary_features = get_all_features(test_texts, vocabulary_binary_vectorizer)

    models = [BernoulliNB(alpha=0.1),
              MultinomialNB(alpha=0.1),
              tree.DecisionTreeClassifier(criterion='entropy', min_samples_leaf=3)]

    train_features_for_models = [train_all_binary_features, train_all_features, train_all_features]
    test_features_for_models = [test_all_binary_features, test_all_features, test_all_features]

    features_nums = 20000
    cv_k = 10

    for model, train_features in zip(models, train_features_for_models):
        perform_cv_for_given_feature_number(model, train_features, train_categories, features_nums, cv_k)

    for model, train_features, test_features in zip(models, train_features_for_models, test_features_for_models):
        evaluate_on_test_data(model, test_features, test_categories, train_features, train_categories, features_nums)


if __name__ == '__main__':
    # run_for_var_feat_num()
    single_run()

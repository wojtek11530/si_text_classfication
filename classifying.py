import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score

from feature_extraction import get_feature_selector, select_features


def evaluate_on_test_data(model, test_all_features, test_categories, train_all_features,
                          train_categories, features_num):
    print('Evaluating model' + type(model).__name__)
    feature_selector = get_feature_selector(train_all_features, train_categories, features_num)
    selected_train_features = select_features(train_all_features, feature_selector)
    selected_test_features = select_features(test_all_features, feature_selector)

    clf = model.fit(selected_train_features, train_categories)
    predict(clf, selected_test_features, test_categories, True)


def predict(model, test_features, test_categories_label, plotting=True):
    model_name = type(model).__name__
    predicted_categories = model.predict(test_features)

    acc = accuracy_score(test_categories_label, predicted_categories)
    conf_mat = confusion_matrix(test_categories_label, predicted_categories)
    if plotting:
        plot_scoring_and_conf_matrix(acc, conf_mat, model, model_name)
    return acc


def plot_scoring_and_conf_matrix(acc, conf_mat, model, model_name):
    plt.subplots(figsize=(14, 9))
    sns.heatmap(conf_mat, annot=True, fmt='d',
                xticklabels=model.classes_, yticklabels=model.classes_, cmap='Blues', cbar=None)
    plt.xticks(
        rotation=45,
        ha='right'
    )
    plt.ylabel('Actual', fontsize=18)
    plt.xlabel('Predicted', fontsize=18)
    plt.title(model_name + ', confusion matrix, accuracy = ' + f'{acc:.5}', fontsize=24)
    plt.tight_layout()
    plt.show()


def perform_cv_for_given_feature_number(model, train_all_features, train_categories, n_feature_num, k=10):
    feature_selector = get_feature_selector(train_all_features, train_categories, n_feature_num)
    selected_features = select_features(train_all_features, feature_selector)
    scores = cross_val_score(model, selected_features, train_categories, cv=k, n_jobs=-1)
    print(type(model).__name__ + " CV, feat num = " + str(n_feature_num) + ", acc.: %0.4f (+/- %0.4f)" % (
        scores.mean(), scores.std()))
    return scores


def perform_cv_for_various_features_num(model, train_all_features, train_categories, features_nums):
    avg_cv_accuracies = []
    cv_k = 10
    print(type(model).__name__ + " cross-validation results, k=" + str(cv_k) + ":")
    for n in features_nums:
        scores = perform_cv_for_given_feature_number(model, train_all_features, train_categories, n, cv_k)
        avg_cv_accuracies.append(np.mean(scores))
    print()
    return avg_cv_accuracies


def determine_the_best_features_num(features_nums, avg_cv_accuracies):
    ind = np.argmax(avg_cv_accuracies)
    n_best = features_nums[ind]
    print("Best features num = " + str(n_best) + ", cv_accuracy: %0.4f" % (avg_cv_accuracies[ind]))
    return n_best


def plot_cv_results(cv_result_tuple):
    for model, features_nums, avg_cv_accuracies in cv_result_tuple:
        plt.plot(features_nums, avg_cv_accuracies, '.--', label=type(model).__name__)

    plt.xlabel(r'number of selected features')
    plt.ylabel('avg accuracy')
    plt.legend()
    plt.grid()
    plt.show()

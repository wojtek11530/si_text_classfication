from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt


def get_multinomial_nb_model(features_data, category_labels):
    clf = MultinomialNB(alpha=1.0).fit(features_data, category_labels)
    return clf


def predict(model, test_features, test_categories_label):
    model_name = type(model).__name__
    predicted_categories = model.predict(test_features)
    report = classification_report(test_categories_label, predicted_categories)

    acc = accuracy_score(test_categories_label, predicted_categories)
    conf_mat = confusion_matrix(test_categories_label, predicted_categories)
    fig, ax = plt.subplots(figsize=(14, 9))
    sns.heatmap(conf_mat, annot=True, fmt='d',
                xticklabels=model.classes_, yticklabels=model.classes_, cmap='Blues', cbar=None)
    plt.xticks(
        rotation=45,
        ha='right'
    );
    plt.ylabel('Actual', fontsize=18)
    plt.xlabel('Predicted', fontsize=18)
    plt.title(model_name + ', confusion matrix, accuracy = ' + f'{acc:.5}', fontsize=24)
    plt.tight_layout()
    plt.show()



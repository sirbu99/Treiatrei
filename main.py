"""
Main module
"""
import pandas as pd

from classifiers.BayesNaiv import Classifier_NB
from classifiers.Random import FlipCoin
from preprocessors.smallgroups import SmallGroups
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import numpy as np
import warnings

# Set to logging to 0 if u dont need logging.
LOGGING = 0


def precision(classifier, preprocessor):
    X, y = preprocessor.get_model()

    # splitting the data set into training set and test set
    conf_matrix = []
    total_test = 0
    total_training = 0
    for i in range(len(X)):
        X_train, X_test, y_train, y_test = train_test_split(
            X[i], y[i], test_size=0.25, random_state=0)

        # identifying what category of messages we need
        needed = 's'
        if "_non-offensive" in classifier.name:
            needed = '0'
        elif "_offensive" in classifier.name:
            needed = '1'

        if needed != 's':
            # train the model
            classifier.fit(X_train, y_train)

            # get the targeted class of messages from the test set
            X_needed = []
            y_needed = []
            last = len(X_test[0]) - 1
            X_needed = [X_test[k] for k in range(len(X_test)) if X_test[k][last] == needed]
            y_needed = [y_test[k] for k in range(len(y_test)) if X_test[k][last] == needed]

            warnings.filterwarnings("ignore")
            # predicting the test set results
            y_pred = classifier.predict(X_needed)
            total_test += len(X_needed)
            total_training += len(X[i])
            cm = confusion_matrix(y_needed, y_pred)
            if len(conf_matrix) == 0:
                conf_matrix = cm
            else:
                for i in range(len(conf_matrix)):
                    for j in range(len(conf_matrix)):
                        conf_matrix[i][j] += cm[i][j]

    # prec = corect_classified / (corect_classified + wrong_classified)
    prec = 0
    if needed == '0':
        prec = conf_matrix[0][0] / (conf_matrix[0][0] + conf_matrix[0][1])
    elif needed == '1':
        prec = conf_matrix[1][1] / (conf_matrix[1][1] + conf_matrix[1][0])

    # recall = 0
    return prec, 0


def precision_recall(classifier, preprocessor):
    _DEBUG = False
    X, y = preprocessor.get_model(_DEBUG)

    # splitting the data set into training set and test set
    conf_matrix = []
    total_test = 0
    total_training = 0
    for i in range(len(X)):
        if _DEBUG:
            X[i] = [(X[i][_], preprocessor.pretty_hot_vector[i][_], preprocessor.dataset[i][_]) for _ in range(len(X[i]))]

        X_train, X_test, y_train, y_test = train_test_split(X[i], y[i], test_size=0.25)

        # DEBUG: am verificat ca datele de la model sunt amestecate de fiecare data cu test_train_split!
        # XDEBUG_train, XDEBUG_test, yDEBUG_train, yDEBUG_test = train_test_split(preprocessor.pretty_hot_vector[i], y[i], test_size=0.25)
        _DEBUG_show_rows = 5
        if _DEBUG:
            XDEBUG_train = [_[1] for _ in X_train]
            XDEBUG_train_review = [_[2] for _ in X_train]
            X_train = [_[0] for _ in X_train]
            XDEBUG_test = [_[1] for _ in X_test]
            XDEBUG_test_review = [_[2] for _ in X_test]
            X_test = [_[0] for _ in X_test]

            print("\n       First %g rows in train set:" %(_DEBUG_show_rows))
            print(XDEBUG_train_review[:_DEBUG_show_rows])
            print(XDEBUG_train[:_DEBUG_show_rows])

            print("\n       First %g rows in test set:" %(_DEBUG_show_rows))
            print(XDEBUG_test_review[:_DEBUG_show_rows])
            print(XDEBUG_test[:_DEBUG_show_rows])

        warnings.filterwarnings("ignore", category=FutureWarning)
        classifier.fit(X_train, y_train)

        # predicting the test set results
        y_pred = classifier.predict(X_test)

        if _DEBUG:
            print("\n       First %g rows in prediction:" %(_DEBUG_show_rows))
            print(y_pred[:_DEBUG_show_rows])

        total_test += len(X_test)
        total_training += len(X_train)
        # print(X_test)

        # making the confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        if len(conf_matrix) == 0:
            conf_matrix = cm
        else:
            for i in range(len(conf_matrix)):
                for j in range(len(conf_matrix)):
                    conf_matrix[i][j] += cm[i][j]

    # print(conf_matrix,total_test,total_training)

    # precision= TP / TP + FP
    prec = conf_matrix[0][0] / (conf_matrix[0][0] + conf_matrix[1][0])
    # print(prec)

    # recall = TP / TP + FN
    recall = conf_matrix[0][0] / (conf_matrix[0][0] + conf_matrix[0][1])
    # print(recall)

    return prec, recall


classifiers = []
preprocessors = []

# (dependency injection) Add new classifiers and/or preprocessors in the lists below
classifiers += [Classifier_NB("BayNv")]
classifiers += [FlipCoin("Flip")]
classifiers += [Classifier_NB("BayNv_offensive")]
classifiers += [Classifier_NB("BayNv_non-offensive")]
preprocessors += [SmallGroups("LemGroups", "data/16k-lemmatized.txt", "data/16k-words-list.txt")]
preprocessors += [SmallGroups("UnprocessedGroups", "data/16k_neprocesat.txt", "data/16k_neprocesat_words_list.txt")]

# compare precision and recall for all combinations of classifers and preprocessors
results = pd.DataFrame(columns=["[" + n1.name + '+' + n2.name + "]" for n1 in classifiers for n2 in preprocessors],
                       index=["precision", "recall"])
for classifier in classifiers:
    for preprocessor in preprocessors:
        print("\n=========================== Processing " + classifier.name + "+" + preprocessor.name + " =====================================")
        prec, recall = 0, 0
        if "_" in classifier.name:
            prec, recall = precision(classifier, preprocessor)
        else:
            prec, recall = precision_recall(classifier, preprocessor)
        results["[" + classifier.name + "+" + preprocessor.name + "]"]["precision"] = prec
        results["[" + classifier.name + "+" + preprocessor.name + "]"]["recall"] = recall
        if LOGGING:
            with open('logs/' + classifier.name + '.log', 'a') as f:  # logging purposes only
                f.write(str(prec) + '\t' + str(recall))
                f.write('\n')
with open("./data/stats.txt", 'a') as f:
    f.write(
        results.to_string()
    )
print(results)

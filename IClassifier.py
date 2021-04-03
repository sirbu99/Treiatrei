"""Interface v0.1

All classifiers should implement the methods in this interface.
Example:

        from IClassifier import *

        class class_name(IClassifier):
            ...
"""

class IClassifier:
    def __init__(self, name: str):
        """Give the classifer a name"""
        self.name = name

    def fit(self, X, Y):
        """Train the classifier.

        Arguments:
        X - train data
        Y - train label

        """
        pass

    def predict(self, X) -> list:
        """Predicts lables.

        Arguments:
        X - test data

        Returns:
        list of predicted labels
        """
        pass
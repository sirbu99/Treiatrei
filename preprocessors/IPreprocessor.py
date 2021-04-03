"""Interface v0.1

Implementation example:

        from IPreprocessor import *

        class class_name(IPreprocessor):
            ...
"""

class IPreprocessor:
    def __init__(self, name: str):
        """Give the preprocessor a name"""
        self.name = name

    def get_model(self, X, Y) -> list:
        """Train the classifier.

        Arguments:
        X - train data
        Y - train label

        """
        pass

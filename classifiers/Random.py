from IClassifier import *
import numpy as np

# TODO implement

class FlipCoin(IClassifier):
    def fit(self, X, Y):
        pass

    def predict(self, X):
	    return np.random.choice(['1', '0'], len(X))
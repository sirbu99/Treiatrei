if __name__ != '__main__':
    
    from sklearn.ensemble import AdaBoostClassifier    
    from IClassifier import *

    class Classifier_AdaBoost(IClassifier):
        def fit(self, X, Y):
            self.classifier = AdaBoostClassifier()
            self.classifier.fit(X, Y)

        def predict(self, X):
            return self.classifier.predict(X)

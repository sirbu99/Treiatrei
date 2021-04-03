if __name__ != '__main__':
    
    from sklearn.naive_bayes import GaussianNB 
    from IClassifier import *

    class Classifier_NB(IClassifier):
        def fit(self, X, Y):
            self.classifier = GaussianNB()
            self.classifier.fit(X, Y)

        def predict(self, X):
            return self.classifier.predict(X)

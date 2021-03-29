"""
Input: corpus-lemmatized.txt, optional corpus-words-list.txt from lematizare.py
"""

# cleaning texts 
import pandas as pd 
import re 
import nltk 
import numpy as np
from nltk.corpus import stopwords 
from nltk.stem.porter import PorterStemmer 
from sklearn.feature_extraction.text import CountVectorizer 

# read files
dataset = []
f_p = open("corpus-lemmatized.txt","r")
for line in f_p.readlines():
	dataset.append([line[2:],line[0]])
f_p.close()

word_count = 4000
words = []
f_n = open("corpus-words-list.txt","r")
word_count = int(f_n.readline())
for line in f_n.readlines():
	words.append([line[2:],line[0]])
f_n.close()


dataset = pd.DataFrame(dataset) 
dataset.columns = ["Text", "Reviews"] 

nltk.download('stopwords') 

corpus = [] 

for i in range(0, len(dataset)): 
	text = re.sub('[^a-zA-Z\s]', '', dataset['Text'][i]) 
	text = text.lower() 
	text = text.split() 
	text = ' '.join(text) 
	corpus.append(text) 

# creating bag of words model 
cv = CountVectorizer(max_features = word_count) 

X = cv.fit_transform(corpus).toarray() 
y = dataset.iloc[:, 1].values 
y_T = y.reshape(len(y),1)

# add prediction as feature
X = np.append(X, y_T, axis=1)

# splitting the data set into training set and test set 
from sklearn.model_selection import train_test_split 

X_train, X_test, y_train, y_test = train_test_split( 
		X, y, test_size = 0.25, random_state = 0) 

# fitting naive bayes to the training set 
from sklearn.naive_bayes import GaussianNB 
from sklearn.metrics import confusion_matrix 

classifier = GaussianNB(); 
classifier.fit(X_train, y_train) 

# predicting test set results 
y_pred = classifier.predict(X_test) 

# making the confusion matrix 
cm = confusion_matrix(y_test, y_pred) 
print(cm) 

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

nltk.download('stopwords') 

# read files
dataset = [[]]
list_index=0
line_index=0
f_p = open("16k-lemmatized.txt","r")
for line in f_p.readlines():
	if line_index == 3500:
		line_index=0
		dataset.append([])
		list_index+=1
	dataset[list_index].append([line[2:],line[0]])
	line_index+=1
f_p.close()


def clasificator(dataset):

	word_count = 4000
	words = []
	f_n = open("16k-words-list.txt","r")
	word_count = int(f_n.readline())
	for line in f_n.readlines():
		words.append([line[2:],line[0]])
	f_n.close()


	dataset = pd.DataFrame(dataset) 
	dataset.columns = ["Text", "Reviews"] 

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
	return X,y

X=[]
y=[]
for i in range(len(dataset)):
	X1,y1=clasificator(dataset[i])
	X.append(X1)
	y.append(y1)


# fitting naive bayes to the training set 
from sklearn.naive_bayes import GaussianNB 
from sklearn.metrics import confusion_matrix 

classifier = GaussianNB()

# splitting the data set into training set and test set 
from sklearn.model_selection import train_test_split 

cm1=[]
total_test=0
total_training=0
for i in range(len(X)):
	X_train, X_test, y_train, y_test = train_test_split( 
			X[i], y[i], test_size = 0.25, random_state = 0) 
	
	classifier.fit(X_train, y_train)

	#predicting the test set results
	y_pred = classifier.predict(X_test)
	total_test+=len(X_test)
	total_training+=len(X_train)
	
	# making the confusion matrix 
	cm = confusion_matrix(y_test, y_pred)
	if len(cm1)==0:
		cm1=cm
	else:
		for i in range(len(cm1)):
			for j in range(len(cm1)):
				cm1[i][j]+=cm[i][j]

print(cm1,total_test,total_training)

prec=cm1[0][0]/(cm1[0][0]+cm1[1][0])
print(prec)

#TP+FN
recall=cm1[0][0]/(cm1[0][0]+cm1[0][1])
print(recall)

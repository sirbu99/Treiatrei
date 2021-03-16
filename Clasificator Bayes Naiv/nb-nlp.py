# cleaning texts 
import pandas as pd 
import re 
import nltk 
from nltk.corpus import stopwords 
from nltk.stem.porter import PorterStemmer 
from sklearn.feature_extraction.text import CountVectorizer 

# read files
positives = []
f_p = open("nonoffensive.txt","r")
for line in f_p.readlines():
	positives.append([line[2:],line[0]])
f_p.close()

negatives = []
f_n = open("offensive.txt","r")
for line in f_n.readlines():
	negatives.append([line[2:],line[0]])
f_n.close()


dataset = positives + negatives
			
dataset = pd.DataFrame(dataset) 
dataset.columns = ["Text", "Reviews"] 

nltk.download('stopwords') 

corpus = [] 

for i in range(0, len(dataset)): 
	text = re.sub('[^a-zA-Z\s]', '', dataset['Text'][i]) 
	text = text.lower() 
	text = text.split() 
	ps = PorterStemmer() 
	text = ' '.join(text) 
	corpus.append(text) 

# creating bag of words model 
cv = CountVectorizer(max_features = 1500) 

X = cv.fit_transform(corpus).toarray() 
y = dataset.iloc[:, 1].values 

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

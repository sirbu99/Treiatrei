if __name__ != '__main__':
	from IPreprocessor import *

	import pandas as pd 
	import re 
	import nltk 
	import numpy as np
	from nltk.corpus import stopwords 
	from nltk.stem.porter import PorterStemmer 
	from sklearn.feature_extraction.text import CountVectorizer 


	class SmallGroups(IPreprocessor):
		def __init__(self, name):
			self.name = name
			# nltk.download('stopwords') 

			# read files and split all messages into smaller groups of mesages
			dataset = [[]] 
			list_index=0 
			line_index=0
			f_p = open("data/16k-lemmatized.txt", "r", encoding='utf-8')
			for line in f_p.readlines():
				if line_index == 3500:
					line_index=0
					dataset.append([])
					list_index+=1
				dataset[list_index].append([line[2:],line[0]])
				line_index+=1
			f_p.close()

			#trecem fiecare lista de mesaje prin fct dataset_filter-> returneaza X,y folositi in antrenarea modelului
			self.X=[]
			self.y=[]
			for i in range(len(dataset)):
				temp_X,temp_y=self.dataset_filter(dataset[i])
				self.X.append(temp_X)
				self.y.append(temp_y)


		def get_model(self):
			return self.X, self.y
			
		def dataset_filter(self, dataset):

			word_count = 4000
			words = []
			f_n = open("data/16k-words-list.txt", "r", encoding='utf-8')
			word_count = int(f_n.readline())
			for line in f_n.readlines():
				words.append([line[2:],line[0]])
			f_n.close()


			dataset = pd.DataFrame(dataset) 
			dataset.columns = ["Text", "Reviews"] 

			corpus = [] 

			for i in range(0, len(dataset)): 
				text = re.sub(r'[^a-zA-Z\s]', '', dataset['Text'][i]) 
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


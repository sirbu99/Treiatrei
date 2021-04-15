if __name__ != '__main__':
	from IPreprocessor import *

	import pandas as pd 
	import re 
	import nltk 
	import numpy as np
	from nltk.corpus import stopwords 
	from nltk.stem.porter import PorterStemmer 
	from sklearn.feature_extraction.text import CountVectorizer
	from os import path


	class SmallGroups(IPreprocessor):
		def __init__(self, name, corpus_path,words_list_path):
			self.name = name
			# nltk.download('stopwords') 

			#check if path are corect and are files
			from os import path
			if not (path.exists(corpus_path) and path.isfile(corpus_path) and path.exists(words_list_path) and path.isfile(words_list_path)):
				raise Exception("Wrong paths:\n"+corpus_path+"\n"+words_list_path)
			else:
				self.corpus_path=corpus_path
				self.words_list_path=words_list_path

			# read files and split all messages into smaller groups of mesages
			self.dataset = [[]] 
			list_index=0 
			line_index=0
			f_p = open(self.corpus_path, "r", encoding='utf-8')
			for line in f_p.readlines():
				if line_index == 3500:
					line_index=0
					self.dataset.append([])
					list_index+=1
				self.dataset[list_index].append([line[2:],line[0]])
				line_index+=1
			f_p.close()

			#trecem fiecare lista de mesaje prin fct dataset_filter-> returneaza X,y folositi in antrenarea modelului
			self.X=[]
			self.y=[]
			for i in range(len(self.dataset)):
				temp_X,temp_y=self.dataset_filter(self.dataset[i])
				self.X.append(temp_X)
				self.y.append(temp_y)


		def get_model(self, DEBUG_MODE = False):
			"""DEBUG_MODE takes more time to generate a visual representation of the one hot vector self.X"""
			self.pretty_hot_vector = []
			for batch in range(len(self.X)):
				self.pretty_hot_vector += [[]]
				if DEBUG_MODE:
					for instance in range(len(self.X[batch])):
						self.pretty_hot_vector[batch] += [[i for i, x in enumerate(self.X[batch][instance]) if x == 1]]

			return self.X, self.y
			
		def dataset_filter(self, subset):

			word_count = 4000
			words = []
			f_n = open(self.words_list_path, "r", encoding='utf-8')
			word_count = int(f_n.readline())
			for line in f_n.readlines():
				words.append([line[2:],line[0]])
			f_n.close()


			subset = pd.DataFrame(subset) 
			subset.columns = ["Text", "Reviews"] 

			corpus = [] 

			for i in range(0, len(subset)): 
				text = re.sub(r'[^a-zA-Z\s]', '', subset['Text'][i]) 
				text = text.lower() 
				text = text.split() 
				text = ' '.join(text) 
				corpus.append(text) 

			# creating bag of words model 
			cv = CountVectorizer(max_features = word_count)

			X = cv.fit_transform(corpus).toarray() 
			y = subset.iloc[:, 1].values 
			y_T = y.reshape(len(y),1)

			# add prediction as feature
			X = np.append(X, y_T, axis=1)
			return X,y


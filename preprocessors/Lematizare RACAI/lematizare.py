"""
Input: text files 'offensive.txt' and 'nonoffensive.txt', coming from the preprocessing function. 
Output: queries RACAI API and stores the lematized instances in a new file
"""
import pandas as pd 
import json
import requests
from sklearn.feature_extraction.text import CountVectorizer 


racai_url = 'http://relate.racai.ro:5000/process'

# read files
positives = []
f_p = open("data/nonoffensive.txt","r")
for line in f_p.readlines():
	positives.append([line[2:],line[0]])
f_p.close()

negatives = []
f_n = open("data/offensive.txt","r")
for line in f_n.readlines():
	negatives.append([line[2:],line[0]])
f_n.close()


#dataset = positives + negatives
dataset=[]
file1=open("data/training_data.txt","r")
for line in file1.readlines():
	dataset.append([line[2:],line[0]])
file1.close()
			
dataset = pd.DataFrame(dataset) 
dataset.columns = ["Text", "Reviews"] 

corpus = [] 
all_words = set() # count all words in corpus

for i in range(0, len(dataset)): 
	racai_query = dataset['Text'][i]

	d = {'text': racai_query,
		'exec': 'lemmatization'}

	fil = requests.post(racai_url, data = d)
	x = json.loads(fil.text)

	if len(x['teprolin-result']['tokenized']) > 0:
		text = [v for i in x['teprolin-result']['tokenized'][0] for (k, v) in i.items() if k == "_lemma"]
		all_words.update(text)

		text = ' '.join(text) 
		corpus.append(text) 

file = open("data/16k-lemmatized.txt", "w", encoding="utf-8")
for i,text in enumerate(corpus):
    file.write(dataset.iloc[i, 1] + ',' + text + '\n')
file.close()

file = open("data/16k-words-list.txt", "w", encoding="utf-8")
file.write(str(len(all_words)) + '\n')
for text in all_words:
    file.write(text + '\n')
file.close()

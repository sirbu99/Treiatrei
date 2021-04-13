"""
Input: A text file, given by path.
Output: Creates file with the words list from the text file.
"""
import pandas as pd 
from sklearn.feature_extraction.text import CountVectorizer 

#dataset = positives + negatives
dataset=[]
file1=open("training_data.txt","r")
for line in file1.readlines():
	dataset.append([line[2:],line[0]])
file1.close()
			
dataset = pd.DataFrame(dataset) 
dataset.columns = ["Text", "Reviews"] 

corpus = [] 
all_words = set() # count all words in corpus

for i in range(len(dataset)):
	words=dataset["Text"][i].split(" ")
	words=[word.replace("\n"," ") for word in words]

	all_words.update(words)

file = open("training_data_words_list.txt", "w", encoding="utf-8")
file.write(str(len(all_words)) + '\n')
for text in all_words:
    file.write(text + '\n')
file.close()

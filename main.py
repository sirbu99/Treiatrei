"""
Main module
"""
import pandas as pd 
from classifiers.BayesNaiv import Classifier_NB
from classifiers.Random import FlipCoin
from preprocessors.smallgroups import SmallGroups
from sklearn.model_selection import train_test_split 
from sklearn.metrics import confusion_matrix
#Set to logging to 0 if u dont need logging.
LOGGING=1

def precision_recall(classifier, preprocessor):
	X, y = preprocessor.get_model()

	# splitting the data set into training set and test set 
	conf_matrix=[]
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
		if len(conf_matrix)==0:
			conf_matrix=cm
		else:
			for i in range(len(conf_matrix)):
				for j in range(len(conf_matrix)):
					conf_matrix[i][j]+=cm[i][j]

	# print(conf_matrix,total_test,total_training)

	# precision= TP / TP + FP
	prec=conf_matrix[0][0]/(conf_matrix[0][0]+conf_matrix[1][0])
	# print(prec)

	# recall = TP / TP + FN
	recall=conf_matrix[0][0]/(conf_matrix[0][0]+conf_matrix[0][1])
	# print(recall)

	return prec, recall
classifiers = []
preprocessors = []

# (dependency injection) Add new classifiers and/or preprocessors in the lists below
classifiers += [Classifier_NB("BayNv")]
classifiers += [FlipCoin("Flip")]
preprocessors += [SmallGroups("LemGroups")]

# compare precision and recall for all combinations of classifers and preprocessors
results = pd.DataFrame(columns = ["[" + n1.name + '+' + n2.name + "]" for n1 in classifiers for n2 in preprocessors], index=["precision", "recall"])
for classifier in classifiers:
	for preprocessor in preprocessors:
		print("Processing " + classifier.name + "+" + preprocessor.name)
		prec, recall = precision_recall(classifier, preprocessor)
		results["[" + classifier.name + "+" + preprocessor.name +"]"]["precision"] = prec
		results["[" + classifier.name + "+" + preprocessor.name + "]"]["recall"] = recall
		if LOGGING:
			with open('logs/'+classifier.name+'.log', 'a') as f:#logging purposes only
				f.write(str(prec)+'\t'+str(recall))
				f.write('\n')

print(results)


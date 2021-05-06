if __name__ != '__main__':
    from IPreprocessor import *
    import pandas as pd
    import re
    import numpy as np
    from sklearn.feature_extraction.text import CountVectorizer
    from os import path

    class SmallGroups(IPreprocessor):
        def __init__(self, name, corpus_path, words_list_path):
            self.name = name
            # nltk.download('stopwords')

            # check if path are corect and are files
            from os import path
            if not (path.exists(corpus_path) and path.isfile(corpus_path) and path.exists(
                    words_list_path) and path.isfile(words_list_path)):
                raise Exception("Wrong paths:\n" + corpus_path + "\n" + words_list_path)
            else:
                self.corpus_path = corpus_path
                self.words_list_path = words_list_path

            # read files and split all messages into smaller groups of mesages
            self.dataset = [[]]
            list_index = 0
            line_index = 0
            f_p = open(self.corpus_path, "r", encoding='utf-8')
            lines = f_p.readlines()

            most_used_words = open('./data/bad_words.txt').readlines()
            self.bad_words = [_.lower().replace('\n', '') for _ in most_used_words]

            for line in lines:
                self.dataset[0].append([line[2:], line[0]])
            f_p.close()
            temp_X, temp_y = self.dataset_filter(self.dataset[0])

            self.X = []
            self.y = []

            # impartim in 10 
            for xi in range(10):
                start_index = xi * int(len(lines) / 10)

                self.X.append(temp_X[start_index:start_index + int(len(lines) / 10)])
                self.y.append(temp_y[start_index:start_index + int(len(lines) / 10)])

        def get_model(self, DEBUG_MODE=False):
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
                words.append([line[2:], line[0]])
            f_n.close()

            subset = pd.DataFrame(subset)
            subset.columns = ["Text", "Reviews"]

            corpus = []
            has_bad_words = subset.iloc[:, 1].values.copy()
            for i in range(0, len(subset)):
                text = re.sub(r'[^a-zA-Z\s]', '', subset['Text'][i])
                text = text.lower()
                text = text.split()

                has_bad_words[i] = '0'
                for wrd in text:
                    if wrd in self.bad_words:
                        has_bad_words[i] = '1'
                        break

                text = ' '.join(text)
                corpus.append(text)

            # creating bag of words model
            cv = CountVectorizer(max_features=word_count)

            X = cv.fit_transform(corpus).toarray()
            self.feature_names = cv.get_feature_names()

            # print(X)
            y = subset.iloc[:, 1].values

            # add bad words presence as feature
            has_bad_words_T = has_bad_words.reshape(len(has_bad_words), 1)
            X = np.append(X, has_bad_words_T, axis=1)

            # # add prediction as feature
            # y_T = y.reshape(len(y), 1)
            # X = np.append(X, y_T, axis=1)

            return X, y


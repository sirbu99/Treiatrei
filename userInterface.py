from tkinter import *
from tkinter import filedialog, messagebox

import numpy as np
import pandas as pd
from classifiers.BayesNaiv import Classifier_NB
from main import classify_message
from preprocessors.smallgroups import SmallGroups
from preprocessors.Preprocesare_text import script_preprocess
import pickle
import os
from sklearn.feature_extraction.text import CountVectorizer


def openFile():
    clearText()
    tf = filedialog.askopenfilename(
        initialdir="/",
        title="Open Text file",
        filetypes=(("Text Files", "*.txt"),)
    )
    pathh.insert(END, tf)
    tf = open(tf)
    data = tf.read()
    txtarea.insert(END, data)
    tf.close()


def clearText():
    txtarea.delete('1.0', END)
    pathh.delete(0, END)
    pathh.insert(0, "")


def showStats():
    novi = Toplevel()
    canvas = Canvas(novi, width=800, height=600)
    canvas.pack(expand=YES, fill=BOTH)
    stats = PhotoImage(file='./data/performance.png')
    canvas.create_image(80, 80, image=stats, anchor=NW)
    canvas.gif1 = stats


def runClassifier():
    model = pickle.load(open('./data/models/BayNv LemGroups.sav', 'rb'))
    messageToBeClassified = txtarea.get("1.0", "end")

    preprocessed_message = re.sub(r'[^a-zA-Z\s]', '', script_preprocess.correctedLine(messageToBeClassified))
    preprocessed_message = preprocessed_message.lower()
    preprocessed_message = preprocessed_message.split()

    most_used_words = open('./data/bad_words.txt').readlines()
    bad_words = [_.lower().replace('\n', '') for _ in most_used_words]

    # remove words not found in corpus
    has_bad_words = False
    only_corpus_words = []
    for wd in preprocessed_message:
        if wd in model['feature names']:
            only_corpus_words += [wd]
            if wd in bad_words and not has_bad_words:
                only_corpus_words += ['hasBadWordsInTheMessage']
                has_bad_words = True

    cv = CountVectorizer(max_features=len(model['feature names']))
    X = cv.fit_transform([' '.join(model['feature names']), ' '.join(only_corpus_words)]).toarray()

    # remove first line
    X = np.delete(X, 0, 0)

    # add columns
    X = np.append(X, np.array(['1' if has_bad_words else '0' for _ in X]).reshape(len(X), 1), axis=1)
    # X = np.append(X, np.array(['2' for _ in X]).reshape(len(X), 1), axis=1)

    pred = model['classifier'].predict(X)
    print('Pred = ' + pred[0])
    if pred[0] == '0':
        messageToDisplay = "The message isn't offensive."
    else:
        messageToDisplay = "The message is offensive!"
    messagebox.showinfo('Result', messageToDisplay)


ws = Tk()
ws.title("Offensive Language Detection for Romanian Language")
ws.geometry("800x600")
ws['bg'] = '#fb0'

label = Label(ws, text="Scrieti un mesaj pentru a fi clasificat", font="Helvetica 16 bold italic", fg="white",
              bg="#fb0")
label.pack()

txtarea = Text(ws, width=80, height=30)
txtarea.pack(pady=20)

pathh = Entry(ws)
pathh.pack(side=LEFT, expand=True, fill=X, padx=20)

Button(ws, text="Show Stats", command=showStats).pack(side=RIGHT, expand=True, fill=X, padx=20)
Button(ws, text="Clear", command=clearText).pack(side=RIGHT, expand=True, fill=X, padx=20)
Button(ws, text="Run", command=runClassifier).pack(side=RIGHT, expand=True, fill=X, padx=20)
Button(ws, text="Open File", command=openFile).pack(side=RIGHT, expand=True, fill=X, padx=20)

ws.mainloop()

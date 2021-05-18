from tkinter import *
from tkinter import filedialog, messagebox, ttk
import tkinter

import numpy as np
import pandas as pd
from classifiers.BayesNaiv import Classifier_NB
from main import classify_message
from preprocessors.smallgroups import SmallGroups
from preprocessors.Preprocesare_text import script_preprocess
import pickle
import os, re
from sklearn.feature_extraction.text import CountVectorizer
from crawlers.yt_crawler import start_crawler, next_yt_comment


def openFile():
    clearText()
    tf = filedialog.askopenfilename(
        initialdir=".",
        title="Open Text file",
        filetypes=(("Text Files", "*.txt"),)
    )
    pathh.insert(END, tf)

    tf = open(tf, "r", encoding='utf-8')
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


def displayResultForHybrid(pred_list):
    result = all(element == pred_list[0] for element in pred_list)
    if result:
        displayResult(pred_list[0])
    else:
        fMeasures = []
        off = 0
        nonOff = 0
        with open("stats_for_hybrid.txt", "r") as f:
            for i in range(0, 3):
                fMeasures.append(f.readline().strip())
        for i in range(0, len(pred_list)):
            if pred_list[i] == '1':
                off += float(fMeasures[i])
                print("off =", off)
            else:
                nonOff += float(fMeasures[i])
                print("nonOff =", nonOff)
        if off > nonOff:
            displayResult('1')
        else:
            displayResult('0')


def displayResult(prediction):
    if prediction == '0':
        messageToDisplay = "The message isn't offensive."
    else:
        messageToDisplay = "The message is offensive!"
    messagebox.showinfo('Result', messageToDisplay)


def runClassifier():
    # luam modelul cu clasificatorul corespunzator
    model = None
    selected_classifier = classifier.get()
    if selected_classifier == "":
        messagebox.showinfo('Warning', 'Please select a classifier!')
    elif selected_classifier == " Hybrid":
        pred_list = []
        model1 = pickle.load(open('./data/models/BayNv LemGroups.sav', 'rb'))
        pred1 = run_model(model1, " Bayes + lemmatization")
        model2 = pickle.load(open('./data/models/Flip LemGroups.sav', 'rb'))
        pred2 = run_model(model2, " Random + lemmatization")
        model3 = pickle.load(open('./data/models/AdaBoost LemGroups.sav', 'rb'))
        pred3 = run_model(model3, " Ada Boost")
        pred_list.extend([pred1, pred2, pred3])
        print(pred_list)
        displayResultForHybrid(pred_list)
        return 0
    elif selected_classifier == " Bayes + lemmatization":
        model = pickle.load(open('./data/models/BayNv LemGroups.sav', 'rb'))
    elif selected_classifier == " Random + lemmatization":
        model = pickle.load(open('./data/models/Flip LemGroups.sav', 'rb'))
    elif selected_classifier == " Random":
        model = pickle.load(open('./data/models/Flip UnprocessedGroups.sav', 'rb'))
    elif selected_classifier == " Bayes":
        model = pickle.load(open('./data/models/BayNv UnprocessedGroups.sav', 'rb'))
    elif selected_classifier == " Ada Boost":
        model = pickle.load(open('./data/models/AdaBoost LemGroups.sav', 'rb'))

    prediction = run_model(model, selected_classifier)
    displayResult(prediction)


def run_model(model, selected_classifier):
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
    if (selected_classifier != ' Ada Boost'):
        X = np.append(X, np.array(['1' if has_bad_words else '0' for _ in X]).reshape(len(X), 1), axis=1)

    pred = model['classifier'].predict(X)
    return pred[0]


def next_comment():
    clearText()
    if url_crawl.get() == '':
        return

    global lasturl
    if url_crawl.get() != lasturl:
        lasturl = url_crawl.get()
        start_crawler(lasturl)

    txtarea.insert(END, next_yt_comment())


lasturl = ""

ws = Tk()
ws.title("Offensive Language Detection for Romanian Language")
ws.geometry("1000x700")
ws['bg'] = '#fb0'

label = Label(ws, text="Scrieti un mesaj pentru a fi clasificat", font="Helvetica 16 bold italic", fg="white",
              bg="#fb0")
label.grid(row=0, column=1)

txtarea = Text(ws, width=80, height=30)
txtarea.grid(row=1, column=1)

pathh = Entry(ws)
pathh.grid(row=3, column=0, pady=2)

url_crawl = Entry(ws)
url_crawl.grid(row=6, column=0, pady=2)

label2 = Label(ws, text="Alegeti clasificatorul", font="Helvetica 16 bold italic", fg="white",
               bg="#fb0")
label2.grid(row=2, column=1, pady=2)

n = tkinter.StringVar()
classifier = ttk.Combobox(ws, width=27, textvariable=n)

classifier['values'] = (' Random + lemmatization',
                        ' Bayes + lemmatization',
                        ' Random',
                        ' Bayes',
                        ' Ada Boost',
                        ' Hybrid'
                        )

classifier.current()
classifier.grid(row=3, column=1, pady=2)
selected_classifier = None

Button(ws, text="Show Stats", command=showStats, height=1, width=20).grid(row=3, column=2, pady=2)
Button(ws, text="Clear", command=clearText, height=1, width=20).grid(row=4, column=2, pady=2)
Button(ws, text="Run", command=runClassifier, height=1, width=20).grid(row=4, column=1, pady=2)
Button(ws, text="Open File", command=openFile, height=1, width=20).grid(row=4, column=0, pady=2, padx=20)

Button(ws, text="Next YT Comment", command=next_comment, height=1, width=20).grid(row=7, column=0, pady=2)

ws.mainloop()

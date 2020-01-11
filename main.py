#Import external packages
import nltk 
from nltk.stem.lancaster import LancasterStemmer 
stemmer = LancasterStemmer() #Stemming reverts a word to it's root meaning, removing unessarry characters and improving the accuracy of the AI

import numpy
import tflearn
import tensorflow
import random
import json

#Data preprocessing loading in all the words and lables, initalizing documents with all of the patterns
with open("intents.json") as file: 
    data = json.load(file)

words = []
labels = []
docs = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        wrds = nltk.word_tokenize(pattern) #Tokenizing compiles each individual word in your string 
        words.extend(wrds)
        docs.append(pattern)
 
    if intent["tag"] not in labels:
        labels.append(intent["tag"])
        


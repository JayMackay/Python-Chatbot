#Import external packages
import nltk 
from nltk.stem.lancaster import LancasterStemmer 
stemmer = LancasterStemmer() #Stemming reverts a word to it's root meaning, removing unnecessary characters and improving the accuracy of the AI

import numpy
import tflearn
import tensorflow
import random
import json

#Data preprocessing loading in all the words and lables, initalizing documents with all of the patterns
with open("intents.json") as file: 
    data = json.load(file)

#Initialize Lists
words = [] 
labels = []
docs_x = [] #List for patterns
docs_y = [] #List for tags

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        wrds = nltk.word_tokenize(pattern) #Tokenizing compiles each individual word in your string 
        words.extend(wrds)
        docs_x.append(wrds)
        docs_y.append(intent["tag"])
 
    if intent["tag"] not in labels:
        labels.append(intent["tag"])

words = [stemmer.stem(w.lower()) for w in words if w != "?"] #Convert elements within "words" list to lower case
words = sorted(list(set(words))) #Remove any duplicate elements and sort it into the list

labels = sorted(labels)

#Implement Bag of Words model for training and testing output
training = []
output = []

out_empty = [0 for _ in range(len(labels))] #Output list for bot responses

for x, doc in enumerate(docs_x):
    bag = []

    wrds = [stemmer.stem(w) for w in doc]  #Stem elements in "docs_x" pattern list

    #One Hot Encoding converting categorical data to numerical values
    for w in words:
        if w in wrds: 
            bag.append(1)
        else:
            bag.append(0)

    #Generate output 
    output_row = out_empty[:]
    output_row[labels.index(docs_y[x])] = 1 #Searches the labels list for the existing tag and sets the value to 1

    training.append(bag) 
    output.append(output_row)

#Convert initialized lists into NumPy arrays
training = numpy.array(training)
output = numpy.array(output)

#Build AI model
tensorflow.reset_default_graph()

net = tflearn.input_data(shape = [None, len(training[0])]) #Define input shape expected for model i.e. the length of the training array
net = tflearn.fully_connected(net, 8) #Add fully connected layer to neural network initializing the input data
net = tflearn.fully_connected(net, 8)
#Softmax provides a probability for each neuron in the layer which gives us the output for the network
net = tflearn.fully_connected(net, len(output[0]), activation = "softmax") 
net = tflearn.regression(net)

model = tflearn.DNN(net)

#Pass training data through the model
model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
model.save("model.tflearn")





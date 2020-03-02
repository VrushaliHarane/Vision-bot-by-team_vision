import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import speech_recognition as sr 
import pyttsx3  
import sys
import time


r = sr.Recognizer() 
def ListenText(): 
    try:
        with sr.Microphone() as source2: 
        
            r.adjust_for_ambient_noise(source2, duration=0.2) 
            audio2 = r.listen(source2) 
            MyText = r.recognize_google(audio2) 
        return MyText
    except:
        print("Could not recognise you. Do you want to speak again?")
        print("type y/n")
        xc=input()
        print("")
        if xc=="y":
            return ListenText()
        else:
            return -1
def SpeakText(command):  
    engine = pyttsx3.init() 
    engine.say(command)  
    engine.runAndWait() 

import numpy
import tflearn
import tensorflow
import random
import json
import pickle

with open("intents.json") as file:
    data = json.load(file)

try:
    
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)
except:
    words = []
    labels = []
    docs_x = []
    docs_y = []

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])

        if intent["tag"] not in labels:
            labels.append(intent["tag"])

    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))

    labels = sorted(labels)

    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []

        wrds = [stemmer.stem(w.lower()) for w in doc]

        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)

        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)


    training = numpy.array(training)
    output = numpy.array(output)

    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)

tensorflow.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

try:
    model.load("model.tflearn")
except:
    model.fit(training, output, n_epoch=10000, batch_size=8, show_metric=True)
    model.save("model.tflearn")

def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
            
    return numpy.array(bag)



with open("intents1.json") as file:
    data = json.load(file)

try:
    xxx
    with open("data1.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)
except:
    words = []
    labels = []
    docs_x = []
    docs_y = []

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])

        if intent["tag"] not in labels:
            labels.append(intent["tag"])

    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))

    labels = sorted(labels)

    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []

        wrds = [stemmer.stem(w.lower()) for w in doc]

        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)

        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)


    training = numpy.array(training)
    output = numpy.array(output)

    with open("data1.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)

tensorflow.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model1 = tflearn.DNN(net)

try:
    xxx
    model1.load("model1.tflearn")
except:
    model1.fit(training, output, n_epoch=10000, batch_size=8, show_metric=True)
    model1.save("model1.tflearn")

def chat():
    print("")
    print("Vision Bot: Hi! I am Vision Bot.(Type 'quit' to stop)!")
    SpeakText("Hi, I am Vision Bot.")
    print("")
    print("Do you want to talk to me or you want to type?")
    print("Type 'y' for talk or 'n' for type")
    x=input()
    print("")
    if x=="y":
        print("Speak on your Microphone to conversate with me")
        while True:
            inp=ListenText()
            if inp==-1:
                print("Vision Bot: Bye! Felt nice to conversate with you.")
                SpeakText("Bye, Felt nice to conversate with you.")
                print("")
                break
            print("User: "+inp)
            print("")
            if inp.lower() == "quit":
                print("Vision Bot: Bye! Felt nice to conversate with you.")
                SpeakText("Bye, Felt nice to conversate with you.")
                print("")
                break

            results = model.predict([bag_of_words(inp, words)])[0]
            results_index = numpy.argmax(results)
            tag = labels[results_index]

            if results[results_index]>0.6:

                for tg in data["intents"]:
                    if tg['tag'] == tag:
                        responses = tg['responses']
                
                yu=random.choice(responses)
                
                print("Vision Bot: ",end="")
                print(yu)
                SpeakText(yu)
                print("")
        
            else:
                
                print("Vision Bot: It was an ambiguous Question! Please Try Again!")
                SpeakText("It was an ambiguous Question! Please Try Again!")
                print("")
    else:
        while True:
            inp = input("User: ")
            print("")
            if inp.lower() == "quit":
                print("Vision Bot: Bye! Felt nice to conversate with you.")
                #SpeakText("Bye! Felt nice to conversate with you.")
                print("")
                break

            results = model.predict([bag_of_words(inp, words)])[0]
            results_index = numpy.argmax(results)
            tag = labels[results_index]

            if results[results_index]>0.6:

                for tg in data["intents"]:
                    if tg['tag'] == tag:
                        responses = tg['responses']
                
                yu=random.choice(responses)
                
                print("Vision Bot: ",end="")
                print(yu)
                #SpeakText(yu)
                print("")
                if tag=="help":
                    print("Chiga")
            else:
                
                print("It was an ambiguous Question! Please Try Again!")
                #SpeakText("Vision Bot: It was an ambiguous Question! Please Try Again!")
                print("")    

chat()
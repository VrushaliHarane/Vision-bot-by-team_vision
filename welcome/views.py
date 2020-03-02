from django.shortcuts import render,redirect
from .models import Product,UserProduct
from django.http import HttpResponse
import math

# Create your views here.
products=Product.objects.all()
    
n=len(products)



import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import speech_recognition as sr 
import pyttsx3  
import sys
import time


r = sr.Recognizer() 
def listentext(): 
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
            return listentext()
        else:
            return -1
def speaktext(command): 
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

with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)

with open("intents1.json") as file:
    data1 = json.load(file)

with open("data1.pickle", "rb") as f:
        words1, labels1, training1, output1 = pickle.load(f)
'''
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
'''
tensorflow.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)
model.load("model.tflearn")
tensorflow.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training1[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output1[0]), activation="softmax")
net = tflearn.regression(net)

model1 = tflearn.DNN(net)
model1.load("model1.tflearn")

'''
except:
    model.fit(training, output, n_epoch=10000, batch_size=8, show_metric=True)
    model.save("model.tflearn")
'''
def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
            
    return numpy.array(bag)
global tagp
tagp=[]
def chat1(request):
    ui=request.GET['name']
    results = model.predict([bag_of_words(ui, words)])[0]
    results_index = numpy.argmax(results)
    tag = labels[results_index]
    tagp.append(tag)
    for tg in data["intents"]:
        if tg['tag'] == tag:
            responses = tg['responses']
    
    yu=random.choice(responses)

    if len(tagp)>=2 and tagp[-2]=="help":
        # url="http://127.0.0.1:8000/search/?usinput="+"+".join(ui.split())
        # # url="http://www.google.com"
        # print(url)
        # print(" ".join(ui.split()))
        

        yu="http://127.0.0.1:8000/search/?usinput="+"+".join(ui.split())
    elif len(tagp)>=2 and tagp[-2]=="order":
        # url="http://127.0.0.1:8000/search/?usinput="+"+".join(ui.split())
        # # url="http://www.google.com"
        # print(url)
        # print(" ".join(ui.split()))
        x=ui.split()
        n1=x[0]
        x=x[1:]
        for i in range(len(x)):
            x[i]=str(x[i])
        yu="http://127.0.0.1:8000/search/?usinput="+"+".join(x)
    return HttpResponse(yu)   
      
'''       
def chatbot():
    
    print("")
    print("Vision Bot: Hi! I am Vision Bot.(Type 'quit' to stop)!")
    speaktext("Hi, I am Vision Bot.")
    print("")
    print("Do you want to talk to me or you want to type?")
    print("Type 'y' for talk or 'n' for type")
    x=input()
    print("")
    if x=="y":
        print("Speak on your Microphone to conversate with me")
        while True:
            inp=listentext()
            if inp==-1:
                print("Vision Bot: Bye! Felt nice to conversate with you.")
                speaktext("Bye, Felt nice to conversate with you.")
                print("")
                break
            print("User: "+inp)
            print("")
            if inp.lower() == "quit":
                print("Vision Bot: Bye! Felt nice to conversate with you.")
                speaktext("Bye, Felt nice to conversate with you.")
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
                speaktext(yu)
                print("")
                
            else:
                
                print("Vision Bot: It was an ambiguous Question! Please Try Again!")
                speaktext("It was an ambiguous Question! Please Try Again!")
                print("")
    else:
        while True:
            inp = input("User: ")
            print("")
            if inp.lower() == "quit":
                print("Vision Bot: Bye! Felt nice to conversate with you.")
                #speaktext("Bye! Felt nice to conversate with you.")
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
                #speaktext(yu)
                print("")
                
            else:
                
                print("It was an ambiguous Question! Please Try Again!")
                #speaktext("Vision Bot: It was an ambiguous Question! Please Try Again!")
                print("")    

'''


def home(request):
    nslides=(n//3)+math.ceil(n/3-n//3)

    params={'no_of_slides':nslides,'range':range(1,nslides), 'product':products}
    return render(request,'new index1.html',params)


def search(request):
    ui=request.GET['usinput']
    xx=[]
    responses=""
    if "".join(ui.split()).isdigit():
        k=[]
        
        for j in ui.split():
            p=[]
            
            for i in products:    
                if int(j) == i.id:
                    p.append(int(j))
                
            if len(p)!=0:
                k.append(set(p))    

        if len(k)!=0:
            k=list(set.union(*k))
    
        p1=Product.objects.filter(pk__in=k)
    
        params={'pro':p1}
        return render(request,'cart.html',params)
    else:
        for i in ui.split():
            results1 = model1.predict([bag_of_words(i, words1)])[0]
            results_index1 = numpy.argmax(results1)
            tag1 = labels1[results_index1]
            if results1[results_index1]>0.6:
                for tg in data1["intents"]:
                    if tg['tag'] == tag1:
                        responses = tg['responses']
            if len(responses)!=0:
                xx.append(random.choice(responses))
            else:
                xx.append(i)
        ui=" ".join(xx)
        k=[]
        for j in ui.lower().split():
            p=[]
            c=1
            for i in products:    
                if j in str(i.id) in i.name.lower().split() or j in i.desc.lower().split() or j in i.tag.lower().split() or j in i.size.lower().split() or j in i.color.lower().split() or j in i.company.lower().split():
                    p.append(c)
                c+=1
            if len(p)!=0:
                k.append(set(p))    
        if len(k)!=0:
            k=list(set.intersection(*k))
        
        p1=Product.objects.filter(pk__in=k)
        
        params={'pro':p1}
        return render(request,'search.html',params)

def shop(request):
    return render(request,'shop.html')

def blog(request):
    return render(request,'blog.html')

def contact(request):
    return render(request,'contact.html')

def blogd(request):
    return render(request,'blog-details.html')

def shoppingcart(request):
    return render(request,'shopping-cart.html')

def checkout(request):
    return render(request,'check-out.html')

def faq(request):
    return render(request,'faq.html')

def register(request):
    return render(request,'register.html')

def login(request):
    return render(request,'login.html')
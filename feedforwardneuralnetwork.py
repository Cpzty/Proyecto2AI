# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 01:40:12 2019

@author: Cristian
"""

import numpy as np
import cv2
import glob
#image_size = 250000
#ahora se automatiza para todas las imagenes
#im = cv2.imread("0.png",0)
#np_im = np.array(im,dtype=float)
#image_size = np_im.shape[0] * np_im.shape[1] 
#np_im = np_im.reshape((1,image_size))
#np_im.reshape()
#print(np_im.shape)
#X = np_im
y = ([1,0,0,0,0,0,0,0,0,0])

#0-circulo
#1-cuadrado
#2-triangulo
#3-huevo
#4-arbol
#5-casa
#6-cara feliz
#7 cara triste
#8 signo interrogacion
#9 mickey mouse

#X = X/np.amax(X,axis=1)

#X= X.reshape(-1,X.shape[1])
#print(y)

def obtain_images():
    image_list=[]
    for filename in glob.glob('circle/*.png'):
        im = cv2.imread(filename,0)
        image_list.append(im)
    return image_list

images = obtain_images()

def compose_X(images):
    X=[]
    for i in range(len(images)):
        X.append(np.array(images[i],dtype=float))
        image_size = X[i].shape[0] * X[i].shape[1] 
        X[i] = X[i].reshape((1,image_size))
        X[i] = X[i]/np.amax(X[i],axis=1)
    return X,image_size

X,image_size = compose_X(images)
#print(X)
#print(image_size)
#y esto varia depende de como se entrene
#y=[]
#for i in range(len(X)):
#    y.append(1)

#print(y)    

class Neural_Network(object):
    def __init__(self):
        self.input_size=image_size
        self.output_size= 10
        self.hidden_size=3
        self.W1 = np.random.randn(self.input_size,self.hidden_size)
        self.W2 = np.random.randn(self.hidden_size,self.output_size)

    def forward(self,X):
        self.z = np.dot(X,self.W1)
        self.z2 = self.sigmoid(self.z)
        self.z3 = np.dot(self.z2,self.W2)
        o = self.sigmoid(self.z3)
        return o
    def sigmoid(self,s):
        return 1/(1 + np.exp(-s))
    
    def sigmoidPrime(self,s):
        return s * (1 - s)
    
    def backward(self,X,y,o):
        self.o_error = y - o
        self.o_delta = self.o_error * self.sigmoidPrime(o)
        self.z2_error = self.o_delta.dot(self.W2.T)
        self.z2_delta = self.z2_error * self.sigmoidPrime(self.z2)
        self.W1 += X.T.dot(self.z2_delta)
        self.W2 += self.z2.T.dot(self.o_delta)
    def train(self,X,y):
        o = self.forward(X)
        self.backward(X,y,o)
    def predict(self):
        print("Data predecida por los pesos entrenados: ")
        print("Input(escalado):{} \n".format(xPredicted))
        print("Salida: {}".format(self.forward(xPredicted)))
    def saveWeights(self):
        np.savetxt("w1.txt",self.W1,fmt="%s")
        np.savetxt("w2.txt",self.W2,fmt="%s")

Neural_forward = Neural_Network()

#o = Neural_forward.forward(X)
#print("prediccion:{}".format(o)) 
#print("real:{}".format(y))

for j in range(100):
    for i in range(10):
        print("input:{}\n".format(X[i]))
        print("output:{}\n".format(y))
        print("prediccion:{}\n".format(Neural_forward.forward(X[i])))
        print("perdida:{}\n".format(np.mean(np.square(y - Neural_forward.forward(X[i])))))
        print("\n")
        Neural_forward.train(X[i],y)
##
#Neural_forward.saveWeights()
xPredicted = X[11]

#xPredicted = xPredicted/np.amax(xPredicted,axis=0)

Neural_forward.predict()


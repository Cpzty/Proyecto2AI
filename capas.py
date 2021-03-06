# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 01:40:12 2019

@author: Cristian
"""

import numpy as np
import cv2
import glob
from random import randint,seed
#image_size = 250000
#ahora se automatiza para todas las imagenes
#im = cv2.imread("0.png",0)
#np_im = np.array(im,dtype=float)
#image_size = np_im.shape[0] * np_im.shape[1] 
#np_im = np_im.reshape((1,image_size))
#np_im.reshape()
#print(np_im.shape)
#X = np_im
y = ([0.1,0,0,0,0,0,0,0,0,0])

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

def decide_category(indx):
    if indx == 0:
        print("Circulo")
    elif indx == 1:
        print("Cuadrado")
    elif indx == 2:
        print("Triangulo")
    elif indx == 3:
        print("Huevo")
    elif indx == 4:
        print("Arbol")
    elif indx == 5:
        print("Casa")
    elif indx == 6:
        print("Cara feliz")
    elif indx == 7:
        print("Cara triste")
    elif indx == 8:
        print("Signo interrogacion")
    elif indx == 9:
        print("Mickey")
        
        
        
        
        
        
        
            

def obtain_images():
    image_list=[]
    for filename in glob.glob('Circle/*.jpg'):
        im = cv2.imread(filename,0)
        image_list.append(im)
    return image_list

def obtain_squares():
    image_list=[]
    for filename in glob.glob('Square/*.bmp'):
        im = cv2.imread(filename,0)
        image_list.append(im)
    return image_list

def obtain_triangles():
    image_list=[]
    for filename in glob.glob('Triangle/*.jpg'):
        im = cv2.imread(filename,0)
        image_list.append(im)
    return image_list

def obtain_eggs():
    image_list=[]
    for filename in glob.glob('Egg/*.bmp'):
        im = cv2.imread(filename,0)
        image_list.append(im)
    return image_list

def obtain_trees():
    image_list=[]
    for filename in glob.glob('Tree/*.jpg'):
        im = cv2.imread(filename,0)
        image_list.append(im)
    return image_list

def obtain_houses():
    image_list=[]
    for filename in glob.glob('House/*.bmp'):
        im = cv2.imread(filename,0)
        image_list.append(im)
    return image_list

def obtain_happy_faces():
    image_list=[]
    for filename in glob.glob('Happy/*.jpg'):
        im = cv2.imread(filename,0)
        image_list.append(im)
    return image_list

def obtain_sad_faces():
    image_list=[]
    for filename in glob.glob('Sad/*.bmp'):
        im = cv2.imread(filename,0)
        image_list.append(im)
    return image_list

def obtain_question():
    image_list=[]
    for filename in glob.glob('Question/*.bmp'):
        im = cv2.imread(filename,0)
        image_list.append(im)
    return image_list

def obtain_mickey():
    image_list=[]
    for filename in glob.glob('Mickey/*.bmp'):
        im = cv2.imread(filename,0)
        image_list.append(im)
    return image_list

def obtain_toPredict():
    image_list=[]
    for filename in glob.glob('Predict/*.bmp'):
        im = cv2.imread(filename,0)
        image_list.append(im)
    return image_list

    

images = obtain_images()

images_squares = obtain_squares()

images_triangles = obtain_triangles()

images_eggs = obtain_eggs()

images_trees = obtain_trees()

images_houses = obtain_houses()

images_happy = obtain_happy_faces()

images_sad = obtain_sad_faces()

images_question = obtain_question()

images_mickey = obtain_mickey()

def compose_X(images):
    X=[]
    for i in range(len(images)):
        X.append(np.array(images[i],dtype=float))
        image_size = X[i].shape[0] * X[i].shape[1] 
        X[i] = X[i].reshape((1,image_size))
        X[i] = X[i]/np.amax(X[i],axis=1)
    return X

X = compose_X(images)
X = np.array(X)
#print(X[])
#print(image_size)
#y esto varia depende de como se entrene
#y=[]
#for i in range(len(X)):
#    y.append(1)

#print(y)    

class Neural_Network(object):
    def __init__(self):
        self.input_size=784
        self.output_size= 10
        self.hidden_size=10
        self.W1 = np.random.randn(self.input_size,self.hidden_size)
        self.W2 = np.random.randn(self.hidden_size,self.hidden_size)
        self.W3 = np.random.randn(self.hidden_size,self.output_size)
        
        

    def forward(self,X):
        print("input:{}\n".format(X))
        self.z = np.dot(X,self.W1)
        self.z2 = self.sigmoid(self.z)
        self.z3 = np.dot(self.z2,self.W2)
        self.z4 = self.sigmoid(self.z3)
        self.z5 = np.dot(self.z4,self.W3)
        o = self.sigmoid(self.z5)
        print("prediccion: {}".format(o))
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
        self.z4_delta = self.z3 * self.sigmoidPrime(self.z3)
        self.W1 += (X.T.dot(self.z2_delta) * 0.001)
        self.W2 += (self.z4.T.dot(self.z4_delta) * 0.001)
        self.W3 += (self.z2.T.dot(self.o_delta) * 0.001)
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

#data
images_squares = obtain_squares()
X2 = compose_X(images_squares)
X2 = np.array(X2)
y2 = [0,0.2,0,0,0,0,0,0,0,0]


X3 = compose_X(images_triangles)
X3 = np.array(X3)
y3 = [0,0,0.3,0,0,0,0,0,0,0]

X4 = compose_X(images_eggs)
#X4 = np.array(X4)
y4 = [0,0,0,0.4,0,0,0,0,0,0]

X5 = compose_X(images_trees)
y5 = [0,0,0,0,0.5,0,0,0,0,0]

X6 = compose_X(images_houses)
y6 = [0,0,0,0,0,0.6,0,0,0,0]

X7 = compose_X(images_happy)
y7 = [0,0,0,0,0,0,0.7,0,0,0]

X8 = compose_X(images_sad)
y8 = [0,0,0,0,0,0,0,0.8,0,0]

X9 = compose_X(images_question)
y9 = [0,0,0,0,0,0,0,0,0.9,0]

X10 = compose_X(images_mickey)
y10 = [0,0,0,0,0,0,0,0,0,1]




#X5, size_image =  

#for j in range(1):
#seed(123)
for i in range(5):
        
        #print("input:{}\n".format(X[i]))
        #print("output:{}\n".format(y))
        #print("prediccion:{}\n".format(Neural_forward.forward(X[i])))
        print("perdida:{}\n".format(np.mean(np.square(y - Neural_forward.forward(X[i])))))
        print("\n")
        try:
            Neural_forward.train(X[randint(0,len(X-1))],y)
            #Neural_forward.train(X[i+1],y)
            Neural_forward.train(X2[randint(0,len(X2)-1)],y2)
            #Neural_forward.train(X2[randint(0,len(X2)-1)],y2)
            Neural_forward.train(X3[randint(0,len(X3)-1)],y3)
            Neural_forward.train(X3[randint(0,len(X3)-1)],y3)
            Neural_forward.train(X4[randint(0,len(X4)-1)],y4)
            Neural_forward.train(X5[randint(0,len(X5)-1)],y5)
            Neural_forward.train(X6[randint(0,len(X6)-1)],y6)
            Neural_forward.train(X7[randint(0,len(X7)-1)],y7)
            Neural_forward.train(X8[randint(0,len(X8)-1)],y8)
            Neural_forward.train(X9[randint(0,len(X9)-1)],y9)
            Neural_forward.train(X10[randint(0,len(X10)-1)],y10)
            if i % 10 == 0:
                Neural_forward.train(X10[randint(0,len(X10)-1)],y10)
        except:
            print("dimension incorrecta")
##
#Neural_forward.saveWeights()
#xPredicted = X2[1100]

#xPredicted = xPredicted/np.amax(xPredicted,axis=0)

#for j in range(1):
 #   for i in range(1000):
  #      print("input:{}\n".format(X2[i]))
   #     print("output:{}\n".format(y2))
     #   print("prediccion:{}\n".format(Neural_forward.forward(X2[i])))
    #    print("perdida:{}\n".format(np.mean(np.square(y2 - Neural_forward.forward(X2[i])))))
      #  print("\n")
 #       Neural_forward.train(X2[i],y2)

predict_images = obtain_toPredict()
X_predict = compose_X(predict_images)
for j in range(len(X_predict)):    
    print("Predicciones")
    #print("imagen#{}".format(j))
    prediccion = Neural_forward.forward(X_predict[j])
    poll = prediccion
    print("imagen#{}".format(j))
    print("creo que pertenece a la categoria:")
    decide_category(np.argmax(poll))
    poll[:,np.argmax(poll)] = 0
    #print(poll)
    print("mi segunda prediccion es la categoria:")
    decide_category(np.argmax(poll))
    poll[:,np.argmax(poll)] = 0
    print("mi tercera prediccion es la categoria:")
    decide_category(np.argmax(poll))

Neural_forward.saveWeights()


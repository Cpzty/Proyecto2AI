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
        self.hidden_size=3
        self.W1 = np.random.randn(self.input_size,self.hidden_size)
        self.W12 = np.random.randn(self.hidden_size,self.hidden_size)
        self.W2 = np.random.randn(self.hidden_size,self.output_size)

    def forward(self,X):
        print("input:{}\n".format(X))
        self.z = np.dot(X,self.W1)
        self.z2 = self.sigmoid(self.z)
        self.z3 = np.dot(self.z2,self.W12)
        self.z4 = self.sigmoid(self.z3)
        self.z5 = np.dot(self.z3,self.W2)
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

#data
images_squares = obtain_squares()
X2 = compose_X(images_squares)
X2 = np.array(X2)
y2 = [0,1,0,0,0,0,0,0,0,0]


X3 = compose_X(images_triangles)
X3 = np.array(X3)
y3 = [0,0,1,0,0,0,0,0,0,0]

X4 = compose_X(images_eggs)
#X4 = np.array(X4)
y4 = [0,0,0,1,0,0,0,0,0,0]

X5 = compose_X(images_trees)
y5 = [0,0,0,0,1,0,0,0,0,0]

X6 = compose_X(images_houses)
y6 = [0,0,0,0,0,1,0,0,0,0]

X7 = compose_X(images_happy)
y7 = [0,0,0,0,0,0,1,0,0,0]

X8 = compose_X(images_sad)
y8 = [0,0,0,0,0,0,0,1,0,0]

X9 = compose_X(images_question)
y9 = [0,0,0,0,0,0,0,0,1,0]

X10 = compose_X(images_mickey)
y10 = [0,0,0,0,0,0,0,0,0,1]




#X5, size_image =  

#for j in range(1):
for i in range(50):
        
        #print("input:{}\n".format(X[i]))
        #print("output:{}\n".format(y))
        #print("prediccion:{}\n".format(Neural_forward.forward(X[i])))
        print("perdida:{}\n".format(np.mean(np.square(y - Neural_forward.forward(X[i])))))
        print("\n")
        Neural_forward.train(X[i],y)
        #Neural_forward.train(X[i+1],y)
        Neural_forward.train(X2[i],y2)
        Neural_forward.train(X3[i],y3)
        Neural_forward.train(X4[i],y4)
        Neural_forward.train(X5[i],y5)
        Neural_forward.train(X6[i],y6)
        Neural_forward.train(X7[i],y7)
        Neural_forward.train(X8[i],y8)
        Neural_forward.train(X9[i],y9)
        Neural_forward.train(X10[i],y10)
        
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

print("Predicciones")
prediccion = Neural_forward.forward(X[52])

poll = prediccion

print("creo que pertenece a la categoria:{}".format(np.argmax(poll)))
poll = np.delete(poll,np.argmax(poll))
#print(poll)
print("mi segunda prediccion es la categoria:{}".format(np.argmax(poll)))
poll = np.delete(poll,np.argmax(poll))
print("mi tercera prediccion es la categoria:{}".format(np.argmax(poll)))


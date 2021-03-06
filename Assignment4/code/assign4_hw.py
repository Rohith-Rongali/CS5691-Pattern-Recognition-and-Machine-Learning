
"""assign4_hw.ipynb

Automatically generated by Colaboratory.

Original file is located at

	https://colab.research.google.com/drive/1v8Lw3DYx0TguMntlIxaEr_esn6xNVAnO?usp=sharing
"""


#pip3 install tensorflow

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import det_curve
from mpl_toolkits import mplot3d
from sklearn.metrics import DetCurveDisplay
import scipy.stats as scs
import statistics
from statistics import mode
from collections import Counter
import os

from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense

tm1 = ['1','3','5','9','z']
tm2 = ['a','ai','bA','chA','lA']


def get_points(dir):
  points = []
  for filename in os.listdir(dir):
    f = open(dir+'/'+filename)
    lines_list = f.readlines()
    arr = np.float_(lines_list[0].split())
    points.append(np.reshape(arr[1:],(int(arr[0]),2)))

  return points


def recenter(points):
  for i in range(len(points)):
    x_centers = (np.amax(points[i][:,0])+np.amin(points[i][:,0]))/2
    y_centers = (np.amax(points[i][:,1])+np.amin(points[i][:,1]))/2
    center = np.vstack((x_centers, y_centers)).T
    points[i] = points[i] - center
    points[i][:,0] = points[i][:,0]/(np.amax(points[i][:,0])-np.amin(points[i][:,0]))
    points[i][:,1] = points[i][:,1]/(np.amax(points[i][:,1])-np.amin(points[i][:,1]))
  return points



def normalise_scores(S):
  return S/np.sum(S,axis=1)[:,np.newaxis]

def ROC(S,groundtruth):

  thresh = np.linspace(np.amin(S),np.amax(S),200)
  #thresh = np.sort(list(set(np.ravel(scores))))
  TPR=[]
  FPR=[]
  FNR=[]
  for t in thresh:
    tp=0
    fp=0
    tn=0
    fn=0

    for i in range(len(groundtruth)):
      for j in range(5):
        if S[i][j]>=t:
          if groundtruth[i] == j:
            tp+=1
          else:
            fp+=1
        else:
          if groundtruth[i] == j:
            fn+=1
          else:
            tn+=1

    TPR.append(tp/(tp+fn))
    FPR.append(fp/(fp+tn))
    FNR.append(fn/(tp+fn))

  return FPR,TPR,FNR

from scipy.signal import resample
from scipy.stats import norm

train_list = []
train_list.append(recenter(get_points('Handwriting_Data/a/train')))
train_list.append(recenter(get_points('Handwriting_Data/ai/train')))
train_list.append(recenter(get_points('Handwriting_Data/bA/train')))
train_list.append(recenter(get_points('Handwriting_Data/chA/train')))
train_list.append(recenter(get_points('Handwriting_Data/lA/train')))

dev_list = []
dev_label = []
for t in range(5):
  temp = recenter(get_points('Handwriting_Data/'+tm2[t]+'/dev'))
  dev_label = dev_label +[t for i in range(len(temp))]
  dev_list = dev_list+temp

"""Resampling to fixed length features"""

mean1 = 0
count = 0
for i in range(5):
  for j in range(len(train_list[i])):
    mean1 = mean1 + len(train_list[i][j])
    count += 1

mean1 = mean1/count

train_res = []
train_label = []
for i in range(5):
  for j in range(len(train_list[i])):
    train_res.append(np.ravel(resample(train_list[i][j], int(mean1))))
    train_label.append(i)

dev_res = []
for i in range(len(dev_list)):
  dev_res.append(np.ravel(resample(dev_list[i], int(mean1))))

train_res = np.array(train_res)
dev_res = np.array(dev_res)





"""PCA"""

cov_matrix2 = np.cov(train_res,rowvar = False)
eigval,eigvec = np.linalg.eigh(cov_matrix2)
order = np.absolute(eigval).argsort()[::-1]
eigval = eigval[order]
eigvec = eigvec[:,order]

plt.figure(figsize=(9,6))
plt.plot(np.absolute(eigval))
plt.title(" Plot of eigenvalues magnitude(descending order) in log-scale",size='x-large')
plt.yscale('log')
plt.xlabel("k")
plt.ylabel("eigenvalues magnitude")
plt.grid()

PC = [eigvec[:,:10],eigvec[:,:20],eigvec[:,:50]]

"""LDA"""

def LDA(X,y,k,plot =False): 
    nf = X.shape[1]
    n= X.shape[0]
    class_labels = np.unique(y)

    S_w = np.zeros((nf, nf),dtype=np.float64)


    S_t =   np.cov(X.T,dtype=np.float64)
        
    for c in class_labels:
      class_items = np.flatnonzero(y == c)
      S_w = S_w + np.cov(X[class_items].T)# * (len(class_items)-1)
        
    S_b = S_t - S_w
    _, eigvec = np.linalg.eigh(np.linalg.pinv(S_w).dot(S_b))
    ldac = eigvec[:,::-1][:,:k]

    tx = X.dot(ldac)

    colors = ['r','g','b','y','k']
    labels = np.unique(y)
    for color, label in zip(colors, labels):
      class_data = tx[np.flatnonzero(y==label)]
      #plt.scatter(class_data[:,0],class_data[:,1],c=color)
    plt.show()

    return ldac


ldac = LDA(train_res,dev_label,4)

"""Code for KNN and LR"""

def knn_pt(pt,train,labels,k): #takes a point assigns a class
  order = np.argsort(np.sum((train-pt)**2,axis=1))
  c = Counter(labels[order][:k])
  return c     #c.most_common(1)[0][0]

def knn(k,train_norm,train_label,test_norm):
  scores = []
  for x in test_norm:
    temp =[]
    c = knn_pt(x,train_norm,train_label,k)
    for i in range(5):
      temp.append(c[i])
    scores.append(temp)
  return scores

"""KNN"""

train_label = np.array(train_label)

scores = []
for k in [5,10,15,20]:
  scores.append(knn(k,train_res,train_label,dev_res))
  print(confusion_matrix(np.argmax(scores[-1],axis=1),dev_label))
m1,m2,m3,m4 = ROC(normalise_scores(scores[0]),dev_label),ROC(normalise_scores(scores[1]),dev_label),ROC(normalise_scores(scores[2]),dev_label),ROC(normalise_scores(scores[3]),dev_label)
plt.figure(figsize=(9,6))
plt.plot(m1[0],m1[1],color='r',label='k=5')
plt.plot(m2[0],m2[1],color='g',label='k=10')
plt.plot(m3[0],m3[1],color='b',label='k=15')
plt.plot(m4[0],m4[1],color='y',label='k=20')
plt.legend()
plt.grid()
plt.ylabel('TPR')
plt.xlabel('FPR')
plt.title("ROC-CURVE")

fig = plt.figure(figsize=(9,6))
ax= fig.gca()
DetCurveDisplay(fpr=m1[0],fnr=m1[2],estimator_name='model-1').plot(ax)
DetCurveDisplay(fpr=m2[0],fnr=m2[2],estimator_name='model-2').plot(ax)
DetCurveDisplay(fpr=m3[0],fnr=m3[2],estimator_name='model-3').plot(ax)
DetCurveDisplay(fpr=m4[0],fnr=m4[2],estimator_name='model-4').plot(ax)
plt.title("DET-CURVE")

scores = []
for i in range(3):
  scores.append(knn(10,train_res @ PC[i],train_label,dev_res @ PC[i]))
  cft = confusion_matrix(np.argmax(scores[-1],axis=1),dev_label)
  print(np.trace(cft)/np.sum(cft))
m1,m2,m3 = ROC(normalise_scores(scores[0]),dev_label),ROC(normalise_scores(scores[1]),dev_label),ROC(normalise_scores(scores[2]),dev_label)
plt.figure(figsize=(9,6))
plt.plot(m1[0],m1[1],color='r',label='k=10')
plt.plot(m2[0],m2[1],color='g',label='k=20')
plt.plot(m3[0],m3[1],color='b',label='k=50')
plt.legend()
plt.grid()
plt.ylabel('TPR')
plt.xlabel('FPR')
plt.title("ROC-CURVE")


fig = plt.figure(figsize=(9,6))
ax= fig.gca()
DetCurveDisplay(fpr=m1[0],fnr=m1[2],estimator_name='k=10').plot(ax)
DetCurveDisplay(fpr=m2[0],fnr=m2[2],estimator_name='k=20').plot(ax)
DetCurveDisplay(fpr=m3[0],fnr=m3[2],estimator_name='k=30').plot(ax)
plt.title("DET-CURVE")

"""## LR"""



def softmax(x):
  #denom = np.sum(np.exp(x),axis=1)
  #return np.exp(x)/denom[:, np.newaxis]
  return (np.exp(x).T / np.sum(np.exp(x),axis=1)).T


def one_hot_enc(Y,nc):
  return np.array([np.eye(1,nc,int(y))[0] for y in Y])

def gradient_descent(X,Y,weight,lr):
  nc = Y.shape[1]
  Y_p = softmax(np.matmul(X,weight))

  N = X.shape[0]

  return (1/N)*np.dot(X.T, (Y_p-Y))

def loss(X,Y,weight): 
  N = X.shape[0]
  return -(np.sum(Y * np.log(softmax(np.matmul(X,weight)))))/N

def logistic_reg(X,Y,lr=0.001,iters=500):
    nf = X.shape[1]
    nc = Y.shape[1]
    N = Y.shape[0]

    loss_log =[]
    epoch = 1

    weight = np.ones((nf, nc),dtype=np.float64)

    while(epoch<iters):

      gradients = gradient_descent(X,Y,weight,lr)
      weight = weight- lr*gradients
      loss_log.append(loss(X,Y,weight))
      
      if (epoch%10 ==0):
        print('iter :'+str(epoch))
        print('Loss :'+str(loss_log[-1]))
        #print(grad)
        print('-----------')

      epoch+=1

    return weight

def predict(X,weight):
  return np.argmax(softmax(X@weight),axis=1),softmax(X@weight)

weight = logistic_reg(train_res,one_hot_enc(train_label,5))
pred1 = predict(dev_res,weight)

cft = confusion_matrix(pred1[0],dev_label)
print(np.trace(cft)/np.sum(cft))
print(cft)

weight = logistic_reg(train_res @ PC[1],one_hot_enc(train_label,5))
pred2 = predict(dev_res@ PC[1],weight)

cft = confusion_matrix(pred2[0],dev_label)
print(np.trace(cft)/np.sum(cft))
print(cft)

weight = logistic_reg(train_res @ ldac,one_hot_enc(train_label,5))
pred3 = predict(dev_res@ ldac,weight)

cft = confusion_matrix(pred3[0],dev_label)
print(np.trace(cft)/np.sum(cft))
print(cft)

scores = pred1[1]

scores2 = pred2[1]
scores3 = pred3[1]

m1,m2,m3 = ROC(normalise_scores(scores),dev_label),ROC(normalise_scores(scores2),dev_label),ROC(normalise_scores(scores3),dev_label)
plt.figure(figsize=(9,6))
plt.plot(m1[0],m1[1],color='r',label='Normal')
plt.plot(m2[0],m2[1],color='g',label='PCA')
#plt.plot(m3[0],m3[1],color='b',label='LDA')
#plt.plot(m4[0],m4[1],color='y',label='k=150')
#plt.plot(m4[0],m4[1],color='k',label='k=200')
plt.legend()
plt.grid()
plt.ylabel('TPR')
plt.xlabel('FPR')
plt.title("ROC-CURVE")


fig = plt.figure(figsize=(9,6))
ax= fig.gca()
DetCurveDisplay(fpr=m1[0],fnr=m1[2],estimator_name='Normal').plot(ax)
DetCurveDisplay(fpr=m2[0],fnr=m2[2],estimator_name='PCA').plot(ax)
#DetCurveDisplay(fpr=m3[0],fnr=m3[2],estimator_name='LDA').plot(ax)

"""## SVM"""

from sklearn import svm
svc = svm.SVC()
from sklearn.model_selection import GridSearchCV

#create a classifier
cls = svm.SVC(gamma=0.1, C=100, kernel="rbf",probability = True)
#train the model
cls.fit(train_res,train_label)
#predict the response
pred = cls.predict_proba(dev_res)
cft = confusion_matrix(np.argmax(pred, axis =1),dev_label)
print(np.trace(cft)/np.sum(cft))

#create a classifier
cls = svm.SVC(gamma=0.1, C=100, kernel="rbf",probability = True)
#train the model
cls.fit(train_res @ PC[1],train_label)
#predict the response
pred2 = cls.predict_proba(dev_res@ PC[1])
cft = confusion_matrix(np.argmax(pred2, axis =1),dev_label)
print(np.trace(cft)/np.sum(cft))
print(cft)

#create a classifier
cls = svm.SVC(gamma=0.1, C=100, kernel="rbf",probability = True)
#train the model
cls.fit(train_res @ ldac,train_label)
#predict the response
pred3 = cls.predict_proba(dev_res@ ldac)
cft = confusion_matrix(np.argmax(pred3, axis =1),dev_label)
print(np.trace(cft)/np.sum(cft))

#please plot the roc for the above three all in one...

scores = pred

scores2 = pred2
scores3 = pred3

m1,m2,m3 = ROC(normalise_scores(scores),dev_label),ROC(normalise_scores(scores2),dev_label),ROC(normalise_scores(scores3),dev_label)
plt.figure(figsize=(9,6))
plt.plot(m1[0],m1[1],color='r',label='Normal')
plt.plot(m2[0],m2[1],color='g',label='PCA')
plt.plot(m3[0],m3[1],color='b',label='LDA')
#plt.plot(m4[0],m4[1],color='y',label='k=150')
#plt.plot(m4[0],m4[1],color='k',label='k=200')
plt.legend()
plt.grid()
plt.ylabel('TPR')
plt.xlabel('FPR')
plt.title("ROC-CURVE")


fig = plt.figure(figsize=(9,6))
ax= fig.gca()
DetCurveDisplay(fpr=m1[0],fnr=m1[2],estimator_name='Normal').plot(ax)
DetCurveDisplay(fpr=m2[0],fnr=m2[2],estimator_name='PCA').plot(ax)
DetCurveDisplay(fpr=m3[0],fnr=m3[2],estimator_name='LDA').plot(ax)







"""## ANN"""



order2 = np.random.permutation(len(train_res))
hw_r = train_res[order2]
hw_label_r = np.array(train_label)[order2]

label_encoder2 = LabelEncoder()
integer_encoded2 = label_encoder2.fit_transform(hw_label_r)


onehot_encoder2 = OneHotEncoder(sparse=False)
integer_encoded2 = integer_encoded2.reshape(len(integer_encoded2), 1)
onehot_encoded2 = onehot_encoder2.fit_transform(integer_encoded2)

classifier_hw = Sequential()
classifier_hw.add(Dense(units = 32, activation = 'relu', input_dim = 152))
classifier_hw.add(Dense(units = 5, activation = 'softmax'))
classifier_hw.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
classifier_hw.fit(hw_r, onehot_encoded2, batch_size = 10, epochs = 100)

y_pred = classifier_hw.predict(dev_res)

confusion_matrix(np.argmax(y_pred,axis=1),dev_label)



classifier_hw = Sequential()
classifier_hw.add(Dense(units = 32, activation = 'relu', input_dim = 20))
classifier_hw.add(Dense(units = 5, activation = 'softmax'))
classifier_hw.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
classifier_hw.fit(hw_r @ PC[1], onehot_encoded2, batch_size = 10, epochs = 100)

y_pred1 = classifier_hw.predict(dev_res @ PC[1])
confusion_matrix(np.argmax(y_pred1,axis=1),dev_label)



classifier_hw = Sequential()
classifier_hw.add(Dense(units = 32, activation = 'relu', input_dim = 4))
classifier_hw.add(Dense(units = 5, activation = 'softmax'))
classifier_hw.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
classifier_hw.fit(hw_r @ ldac, onehot_encoded2, batch_size = 10, epochs = 100)

y_pred2 = classifier_hw.predict(dev_res @ ldac)
confusion_matrix(np.argmax(y_pred2,axis=1),dev_label)

scores = y_pred

scores2 = y_pred1
scores3 = y_pred2

m1,m2,m3 = ROC(normalise_scores(scores),dev_label),ROC(normalise_scores(scores2),dev_label),ROC(normalise_scores(scores3),dev_label)
plt.figure(figsize=(9,6))
plt.plot(m1[0],m1[1],color='r',label='Normal')
plt.plot(m2[0],m2[1],color='g',label='PCA')
plt.plot(m3[0],m3[1],color='b',label='LDA')
#plt.plot(m4[0],m4[1],color='y',label='k=150')
#plt.plot(m4[0],m4[1],color='k',label='k=200')
plt.legend()
plt.grid()
plt.ylabel('TPR')
plt.xlabel('FPR')
plt.title("ROC-CURVE")


fig = plt.figure(figsize=(9,6))
ax= fig.gca()
DetCurveDisplay(fpr=m1[0],fnr=m1[2],estimator_name='Normal').plot(ax)
DetCurveDisplay(fpr=m2[0],fnr=m2[2],estimator_name='PCA').plot(ax)
DetCurveDisplay(fpr=m3[0],fnr=m3[2],estimator_name='LDA').plot(ax)




plt.show()

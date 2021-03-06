# -*- coding: utf-8 -*-
"""assign4_sp.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1C6Nk28GIJ8lPbM2kcia0IWyg8GBWZM1c
"""



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
from scipy.signal import resample
from scipy.stats import norm

from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense

from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

"""#KNN"""

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

tm1 = ['1','3','5','9','z']
tm2 = ['a','ai','bA','chA','lA']

#Function to get all files
def get_template(dir):
  file_iter=0
  template = []
  for filename in os.listdir(dir):
    if filename[-4:] == 'mfcc':
      f = open(dir+'/'+filename)
      lines_list = f.readlines()
      l = np.int_(lines_list[0].split())[1]
      template.append([ np.float_(lines_list[i+1].split())   for i in range(l)  ]  )
  return template


train_list=[]
train_list.append(get_template('Isolated_Digits/'+tm1[0]+'/train'))
train_list.append(get_template('Isolated_Digits/'+tm1[1]+'/train'))
train_list.append(get_template('Isolated_Digits/'+tm1[2]+'/train'))
train_list.append(get_template('Isolated_Digits/'+tm1[3]+'/train'))
train_list.append(get_template('Isolated_Digits/'+tm1[4]+'/train'))

dev_list=[]
dev_label = []
for t in range(5):
  temp = get_template('Isolated_Digits/'+tm1[t]+'/dev')
  dev_label= dev_label +[t for i in range(len(temp))]
  dev_list = dev_list+temp

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

"""Normalising data"""

train_norm = (train_res - np.mean(train_res, axis=0)) / np.std(train_res, axis=0)
test_norm = (dev_res- np.mean(train_res, axis=0)) / np.std(train_res, axis=0)

"""## PCA"""

cov_matrix2 = np.cov(train_norm,rowvar = False)
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

PC = [eigvec[:,:4],eigvec[:,:50],eigvec[:,:100],eigvec[:,:150],eigvec[:,:200]]

def LDA(X,y,k,plot =False): 
    nf = X.shape[1]
    n= X.shape[0]
    class_labels = np.unique(y)

    S_w = np.zeros((nf, nf),dtype=np.float64)


    S_t =   np.cov(X.T,dtype=np.float64)
        
    for c in class_labels:
      class_items = np.flatnonzero(y == c)
      S_w = S_w + np.cov(X[class_items].T) #* (len(class_items)-1)
        
    S_b = S_t - S_w
    _, eigvec = np.linalg.eigh(np.linalg.pinv(S_w).dot(S_b))
    ldac = eigvec[:,::-1][:,:k]

    tx = X.dot(ldac)

    colors = ['r','g','b','y','k']
    labels = np.unique(y)
    for color, label in zip(colors, labels):
      class_data = tx[np.flatnonzero(y==label)]
      #plt.scatter(class_data[:,0],class_data[:,1],c=color)
    #plt.show()

    return ldac


ldac = LDA(train_norm,dev_label,4)



"""## Code for KNN and Logistic regression"""

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

"""## KNN"""

train_label = np.array(train_label)

scores = []
for k in [5,10,15,20]:
  scores.append(knn(k,train_norm,train_label,test_norm))
  cft = confusion_matrix(np.argmax(scores[-1],axis=1),dev_label)
  print(np.trace(cft)/np.sum(cft))
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
for i in range(5):
  scores.append(knn(10,train_norm @ (PC[i]),train_label,test_norm @ (PC[i])))
  cft = confusion_matrix(np.argmax(scores[-1],axis=1),dev_label)
  print(np.trace(cft)/np.sum(cft))
m1,m2,m3,m4,m5 = ROC(normalise_scores(scores[0]),dev_label),ROC(normalise_scores(scores[1]),dev_label),ROC(normalise_scores(scores[2]),dev_label),ROC(normalise_scores(scores[3]),dev_label),ROC(normalise_scores(scores[4]),dev_label)
plt.figure(figsize=(9,6))
plt.plot(m1[0],m1[1],color='r',label='k=25')
plt.plot(m2[0],m2[1],color='g',label='k=50')
plt.plot(m3[0],m3[1],color='b',label='k=100')
plt.plot(m4[0],m4[1],color='y',label='k=150')
plt.plot(m4[0],m4[1],color='k',label='k=200')
plt.legend()
plt.grid()
plt.ylabel('TPR')
plt.xlabel('FPR')
plt.title("ROC-CURVE")


fig = plt.figure(figsize=(9,6))
ax= fig.gca()
DetCurveDisplay(fpr=m1[0],fnr=m1[2],estimator_name='k=25').plot(ax)
DetCurveDisplay(fpr=m2[0],fnr=m2[2],estimator_name='k=50').plot(ax)
DetCurveDisplay(fpr=m3[0],fnr=m3[2],estimator_name='k=100').plot(ax)
DetCurveDisplay(fpr=m4[0],fnr=m4[2],estimator_name='k=150').plot(ax)
DetCurveDisplay(fpr=m5[0],fnr=m5[2],estimator_name='k=200').plot(ax)
plt.title("DET-CURVE")

cft = confusion_matrix(np.argmax(knn(10,train_norm @ (ldac),train_label,test_norm @ (ldac)),axis=1),dev_label)
print(np.trace(cft)/np.sum(cft))
print(cft)

"""# Logistic reg"""

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

weight = logistic_reg(train_norm,one_hot_enc(train_label,5))
pred = predict(test_norm,weight)

cft = confusion_matrix(pred[0],dev_label)
print(np.trace(cft)/np.sum(cft))
print(cft)



weight = logistic_reg(train_norm@PC[3],one_hot_enc(train_label,5))
pred1 = predict(test_norm@PC[3],weight)

cft = confusion_matrix(pred1[0],dev_label)
print(np.trace(cft)/np.sum(cft))
print(cft)



weight = logistic_reg(train_norm@ldac,one_hot_enc(train_label,5))
pred2 = predict(test_norm@ldac,weight)

cft = confusion_matrix(pred2[0],dev_label)
print(np.trace(cft)/np.sum(cft))
print(cft)



m1,m2,m3 = ROC(normalise_scores(pred[1]),dev_label),ROC(normalise_scores(pred1[1]),dev_label),ROC(normalise_scores(pred2[1]),dev_label)
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
#DetCurveDisplay(fpr=m4[0],fnr=m4[2],estimator_name='model-4').plot(ax)
#DetCurveDisplay(fpr=m5[0],fnr=m5[2],estimator_name='model-4').plot(ax)
plt.title("DET-CURVE")



"""#SVM"""

from sklearn import svm
svc = svm.SVC()
from sklearn.model_selection import GridSearchCV

#create a classifier
cls = svm.SVC(gamma=0.1, C=100, kernel="rbf", probability = True)

#train the model
#With PCA
cls.fit(train_norm @ (PC[2]),train_label)
#predict the response
pred = cls.predict_proba(test_norm @ (PC[2]))
pred1 = cls.predict(test_norm @ (PC[2]))
cft = confusion_matrix(pred1,dev_label)
print(np.trace(cft)/np.sum(cft))
print(cft)

#train the model
cls.fit(train_norm ,train_label)
#predict the response
pred = cls.predict_proba(test_norm)
pred1 = cls.predict(test_norm)
cft = confusion_matrix(pred1,dev_label)
print(np.trace(cft)/np.sum(cft))
print(cft)

#With LDA
cls.fit(train_norm @ (ldac),train_label)
#predict the response
pred = cls.predict_proba(test_norm @ (ldac))
pred1 = cls.predict(test_norm @ (ldac))
cft = confusion_matrix(pred1,dev_label)
print(np.trace(cft)/np.sum(cft))
print(cft)

"""#ANN

"""

orderdig = np.random.permutation(len(train_norm))
dig_r = train_norm[orderdig]
dig_label_r = np.array(train_label)[orderdig]

label_encoderdig = LabelEncoder()
integer_encodeddig = label_encoderdig.fit_transform(dig_label_r)
#print(integer_encodeddig)

onehot_encoderdig = OneHotEncoder(sparse=False)
integer_encodeddig = integer_encodeddig.reshape(len(integer_encodeddig), 1)
onehot_encodeddig = onehot_encoderdig.fit_transform(integer_encodeddig)

classifier_dig = Sequential()
classifier_dig.add(Dense(units = 512, activation = 'relu', input_dim = train_norm.shape[1]))
classifier_dig.add(Dense(units = 128, activation = 'relu'))
classifier_dig.add(Dense(units = 64, activation = 'relu'))
classifier_dig.add(Dense(units = 5, activation = 'softmax'))
classifier_dig.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

classifier_dig.fit(dig_r, onehot_encodeddig, batch_size = 10, epochs = 100)

y_preddig = classifier_dig.predict(test_norm)

cft = confusion_matrix(np.argmax(y_preddig,axis=1),dev_label)
print(np.trace(cft)/np.sum(cft))
print(cft)

classifier_dig = Sequential()
classifier_dig.add(Dense(units = 64, activation = 'relu', input_dim = 100))
classifier_dig.add(Dense(units = 32, activation = 'relu'))
classifier_dig.add(Dense(units = 5, activation = 'softmax'))
classifier_dig.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

classifier_dig.fit(dig_r @ PC[2], onehot_encodeddig, batch_size = 10, epochs = 100)

y_preddig2 = classifier_dig.predict(test_norm @ PC[2])

cft = confusion_matrix(np.argmax(y_preddig2,axis=1),dev_label)
print(np.trace(cft)/np.sum(cft))
print(cft)



classifier_dig = Sequential()
classifier_dig.add(Dense(units = 64, activation = 'relu', input_dim = 150))
classifier_dig.add(Dense(units = 32, activation = 'relu'))
classifier_dig.add(Dense(units = 5, activation = 'softmax'))
classifier_dig.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

classifier_dig.fit(dig_r @ PC[3], onehot_encodeddig, batch_size = 10, epochs = 100)

y_preddig3 = classifier_dig.predict(test_norm @ PC[3])

cft = confusion_matrix(np.argmax(y_preddig3,axis=1),dev_label)
print(np.trace(cft)/np.sum(cft))
print(cft)





classifier_dig = Sequential()
classifier_dig.add(Dense(units = 64, activation = 'relu', input_dim = 4))
classifier_dig.add(Dense(units = 32, activation = 'relu'))
classifier_dig.add(Dense(units = 5, activation = 'softmax'))
classifier_dig.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

classifier_dig.fit(dig_r @ ldac, onehot_encodeddig, batch_size = 10, epochs = 100)

y_preddig3 = classifier_dig.predict(test_norm @ ldac)

cft = confusion_matrix(np.argmax(y_preddig3,axis=1),dev_label)
print(np.trace(cft)/np.sum(cft))
print(cft)

25#please plot the roc for above three or four..

# scores = []
# for i in range(5):
#   scores.append(knn(10,train_norm @ (PC[i]),train_label,test_norm @ (PC[i])))
#   cft = confusion_matrix(np.argmax(scores[-1],axis=1),dev_label)
#   print(np.trace(cft)/np.sum(cft))


scores = y_preddig

scores2 = y_preddig2
scores3 = y_preddig3


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
#DetCurveDisplay(fpr=m4[0],fnr=m4[2],estimator_name='model-4').plot(ax)
#DetCurveDisplay(fpr=m5[0],fnr=m5[2],estimator_name='model-4').plot(ax)
plt.title("DET-CURVE")

plt.show()


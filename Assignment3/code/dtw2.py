tm1 = ['1','3','5','9','z']
tm2 = ['a','ai','bA','chA','lA']


#! pip3 install numba   #to speed-up dtw please install numba

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import det_curve
from mpl_toolkits import mplot3d
from sklearn.metrics import DetCurveDisplay


def get_points(dir):
  points = []
  for filename in os.listdir(dir):
    f = open(dir+'/'+filename)
    lines_list = f.readlines()
    arr = np.float_(lines_list[0].split())
    points.append(np.reshape(arr[1:],(int(arr[0]),2)))

  return points

#function to recenter and normalise the digits
def recenter(points):
  for i in range(len(points)):
    x_centers = (np.amax(points[i][:,0])+np.amin(points[i][:,0]))/2
    y_centers = (np.amax(points[i][:,1])+np.amin(points[i][:,1]))/2
    center = np.vstack((x_centers, y_centers)).T
    points[i] = points[i] - center
    points[i][:,0] = points[i][:,0]/(np.amax(points[i][:,0])-np.amin(points[i][:,0]))
    points[i][:,1] = points[i][:,1]/(np.amax(points[i][:,1])-np.amin(points[i][:,1]))
  return points

def plot_points(points):
  plt.figure(figsize=(20,20))
  for n in range(len(points)) : 
      plt.subplot(10,7,n+1)
      plt.plot(points[n][:,0],points[n][:,1]) 


from numba import njit

@njit
def dtw(s, t):
    n, m = len(s), len(t)

    
    dtw_matrix = np.zeros((n+1, m+1))
    
    dtw_matrix.fill(np.inf)

    dtw_matrix[0, 0] = 0.0
    
    for i in range(1, n+1):
        for j in range(1,m+1):
            cost = np.linalg.norm(s[i-1] - t[j-1])         
            dtw_matrix[i, j] = cost + min(min(dtw_matrix[i-1, j], dtw_matrix[i, j-1]), dtw_matrix[i-1, j-1])
    return dtw_matrix[n,m]




#plot one charater and display the sequence in which it's written
p1 = recenter(get_points('Handwriting_Data/a/train'))[0]
plt.figure(figsize=(8,8))
i=1
for n in range(0,len(p1),10) : 
  plt.subplot(3,3,i)
  plt.plot(p1[:n][:,0],p1[:n][:,1])
  plt.xlim(-0.55,0.55)
  plt.ylim(-0.55,0.55)
  i+=1


template_list2 = []
template_list2.append(recenter(get_points('Handwriting_Data/a/train')))
template_list2.append(recenter(get_points('Handwriting_Data/ai/train')))
template_list2.append(recenter(get_points('Handwriting_Data/bA/train')))
template_list2.append(recenter(get_points('Handwriting_Data/chA/train')))
template_list2.append(recenter(get_points('Handwriting_Data/lA/train')))


#plot_points(template_list2[0])
#plot_points(template_list2[1])
#plot_points(template_list2[2])
#plot_points(template_list2[3])
#plot_points(template_list2[4])
#uncomment the above if u wanna see the characters


dev_list2 = []
dev_label2 = []
for t in range(5):
  temp = recenter(get_points('Handwriting_Data/'+tm2[t]+'/dev'))
  dev_label2 = dev_label2 +[t for i in range(len(temp))]
  dev_list2 = dev_list2+temp



def ROC(scores,dev_label):
  #S = np.zeros((len(dev_tb),3))
  #for i in range(len(dev_tb)):
  #  for j in range(3):

  thresh = np.linspace(np.amin(scores),np.amax(scores),200)
  TPR=[]
  FPR=[]
  FNR=[]
  for t in thresh:
    tp=0
    fp=0
    tn=0
    fn=0

    for i in range(len(dev_label)):
      for j in range(5):
        if scores[i][j]>=t:
          if dev_label[i] == j:
            tp+=1
          else:
            fp+=1
        else:
          if dev_label[i] == j:
            fn+=1
          else:
            tn+=1

    TPR.append(tp/(tp+fn))
    FPR.append(fp/(fp+tn))
    FNR.append(fn/(tp+fn))

  return FPR,TPR,FNR



scores2=[]
pred_list2=np.zeros(len(dev_list2))
for i in range(len(dev_list2)):
  scores2.append([np.mean(np.sort([dtw(dev_list2[i],template_list2[j][k]) for k in range(len(template_list2[j])) ])[:15] )   for j in range(5) ])
  pred_list2[i] = np.argmin(scores2[-1])


print(confusion_matrix(pred_list2,dev_label2))

norm_scores2 = [1-i/np.sum(i) for i in scores2]


plt.figure()

plt.plot(ROC(norm_scores2,dev_label2)[0],ROC(norm_scores2,dev_label2)[1],color='k')
plt.grid()
plt.ylabel('TPR')
plt.xlabel('FPR')
plt.title("ROC-CURVE")


m1 = ROC(norm_scores2,dev_label2)
fig = plt.figure(figsize=(7,5))
ax= fig.gca()
DetCurveDisplay(fpr=m1[0],fnr=m1[2]).plot(ax)


plt.show()










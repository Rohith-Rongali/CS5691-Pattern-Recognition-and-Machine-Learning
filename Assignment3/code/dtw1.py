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



#Function to get all files
def get_template(dir):
  file_iter=0
  template = []
  for filename in os.listdir(dir):
    if filename[-4:] == 'mfcc':
      f = open(dir+'/'+filename)
      lines_list = f.readlines()
      l = np.int_(lines_list[0].split())[1]
      template.append(np.array([ np.float_(lines_list[i+1].split())   for i in range(l)  ])  )
  return template


template_list=[]
template_list.append(get_template('Isolated_Digits/'+tm1[0]+'/train'))
template_list.append(get_template('Isolated_Digits/'+tm1[1]+'/train'))
template_list.append(get_template('Isolated_Digits/'+tm1[2]+'/train'))
template_list.append(get_template('Isolated_Digits/'+tm1[3]+'/train'))
template_list.append(get_template('Isolated_Digits/'+tm1[4]+'/train'))


dev_list=[]
dev_label = []
for t in range(5):
  temp = get_template('Isolated_Digits/'+tm1[t]+'/dev')
  dev_label= dev_label +[t for i in range(len(temp))]
  dev_list = dev_list+temp



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




scores=[]
pred_list=np.zeros(len(dev_list))
for i in range(len(dev_list)):
  scores.append([np.mean(np.sort([dtw(dev_list[i],template_list[j][k]) for k in range(len(template_list[j])) ])[:10])   for j in range(5) ])
  pred_list[i] = np.argmin(scores[-1])




print(confusion_matrix(pred_list,dev_label))


norm_scores = [1-i/np.sum(i) for i in scores]



plt.plot(ROC(norm_scores,dev_label)[0],ROC(norm_scores,dev_label)[1],color='k')
#lt.legend()
plt.grid()
plt.ylabel('TPR')
plt.xlabel('FPR')
plt.title("ROC-CURVE")


m1 = ROC(norm_scores,dev_label)
fig = plt.figure(figsize=(7,5))
ax= fig.gca()
DetCurveDisplay(fpr=m1[0],fnr=m1[2]).plot(ax)



plt.show()






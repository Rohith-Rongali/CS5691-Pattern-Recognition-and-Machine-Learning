import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import det_curve
from mpl_toolkits import mplot3d
from sklearn.metrics import DetCurveDisplay
import scipy.stats as scs
import os
from numba import njit

imgs = ['coast','forest','highway','mountain','opencountry']



def get_train(dir):
    file_iter=0
    train = []
    for filename in os.listdir(dir):
        maxs=[]
        mins=[]
        if filename != '.ipynb_checkpoints':
            f = open(dir+'/'+filename)
            lines_list = f.readlines()
            temp = np.array([ np.array(lines_list[i].split(),dtype=np.float64)   for i in range(len(lines_list))  ],dtype=np.float64)
            train.append(temp)
     
    return train


train = []
for i in range(5):
    dir  = 'Features/'+imgs[i]+'/train'
    train.append(get_train(dir))

dev =[]
dev_label = []
for t in range(5):
  temp = get_train('Features/'+imgs[t]+'/dev')
  dev_label= dev_label +[t for i in range(len(temp))]
  dev = dev + temp


images = []
for k in range(len(train)):
    blocks = []
    for i in range(len(train[k])):
        for j in range(len(train[k][i])):
            blocks.append(train[k][i][j])
    blocks = np.array(blocks)
    images.append(blocks)





np.random.seed(42)
def generate(data_set, k):
    centers = []
    index = np.random.choice(range(len(data_set)),k)
    for t in index:
        centers.append(data_set[t])
    return centers


def pt_assign(centers,data):
    labels = []
    for pt in data:
        labels.append(np.argmin([ np.linalg.norm(pt-cen) for cen in centers ]))
    return labels

def update_centers(old_centers,data,labels,k):
    new_centers=[]
    df = pd.DataFrame(data)
    df['labels'] = labels
    for i in range(k):
        if len(df.loc[df['labels']== i]) == 0 :
            new_centers.append(old_centers[i])       
        else:
            new_centers.append(np.mean(df.loc[df['labels']== i].values[:,:-1],axis=0))
        #print(df.loc[df['labels']== i].values)
    return new_centers
        
def kmeans(k,data):
    new_centers = generate(data,k)
    old_centers = [np.zeros(len(new_centers[0]))  for i in range(len(new_centers))   ]
    #while np.array(new_centers).any() != np.array(old_centers).any():
    while np.linalg.norm(np.array(new_centers)-np.array(old_centers))>0.1:
        #print(new_centers)
        old_centers = new_centers
        labels = pt_assign(old_centers,data)
        #print(labels)
        new_centers = update_centers(old_centers,data, labels, k)
        #print(new_centers)
        #print(np.linalg.norm(np.array(new_centers)-np.array(old_centers)))
    return  labels,new_centers



@njit
def mvn(x,mean,cov):
    return ((2*np.pi)**(-len(x)/2))*(np.linalg.det(cov)**(-0.5))*np.exp(-(x-mean).T @ np.linalg.pinv(cov) @ (x-mean)/2)


def mat_cal(data, K, pi, mean, cov):
  mat = np.zeros((len(data), K),dtype=np.float64)
  denom = np.zeros(len(data),dtype=np.float64)
 
  for k in range (K):
    for i in range(len(data)):
      mat[i][k] = pi[k]*mvn(data[i],mean[k],cov[k])
   
  mat = mat/np.tile(np.sum(mat,axis=1),(K,1)).T
 
  return mat

def m_step_new(data, mat, pi,mean,cov):
  N = len(data)
  #data = class1[['x','y']].values
  pi_new = np.array([ np.sum([ mat[i][k]/N for i in range(len(data)) ]) for k in range(K) ])
  mean_new = np.array([ np.sum([ (mat[i][k] * data[i])/(N*pi_new[k])  for i in range(len(data))],axis=0) for k in range(K) ])
  cov_new = np.array([ np.sum([ (mat[i][k] * np.outer(data[i]-mean_new[k],data[i]-mean_new[k]))/(N*pi_new[k])  for i in range(len(data))],axis=0) for k in range(K) ])
  return pi_new,mean_new,cov_new

def gmm(data,mean_i,cov_i,pi_i,iters=10):
  K=len(pi_i)
  pi,mean,cov = pi_i,mean_i,cov_i
  for i in range(iters):
    pi_old =pi
    mean_old = mean
    cov_old = cov
    matri = mat_cal(data, K, pi_old, mean_old, cov_old)
    pi,mean,cov = m_step_new(data, matri, pi_old, mean_old, cov_old)
  return mean,cov,pi




K = 30
prob = []
for r in range(5):
    data = images[r]
    res = kmeans(K,data)
    mean_i = np.array(res[1])
    labels= np.array(res[0])


    cov_i = np.array([ np.sum([np.outer(x-mean_i[i],x-mean_i[i]) for x in data[labels==i]],axis=0)/len(data[labels==i]) for i in range(K) ])
    pi_i = np.array([ np.count_nonzero(data==i)/len(data)  for i in range(K)])

    means,covs,pis = gmm(data,mean_i,cov_i,pi_i,iters=10)
    
    prob_class = np.zeros(len(dev),dtype=np.float64)
    
    #taking log-likelihood across all blocks
    for i in range(len(dev)):
        prob_class[i] = np.sum([np.log(np.sum([mvn(dev[i][q],means[p],covs[p])*pis[p] for p in range(K)])) for q in range(36)])
        
    prob.append(prob_class)


scores = np.array(prob).T    #list of scores

pred_label = np.argmax(scores,axis=1)


print(confusion_matrix(dev_label,pred_label))



def ROC(scores,dev_label):
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


plt.figure()

plt.plot(ROC(scores,dev_label)[0],ROC(scores,dev_label)[1],color='k')
plt.grid()
plt.ylabel('TPR')
plt.xlabel('FPR')
plt.title("ROC-CURVE")


m1 = ROC(scores,dev_label)
fig = plt.figure(figsize=(7,5))
ax= fig.gca()
DetCurveDisplay(fpr=m1[0],fnr=m1[2]).plot(ax)


plt.show()














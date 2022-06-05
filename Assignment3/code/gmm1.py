'''
Team 21
Submssion by: Rongali Rohith (EE19B114) and Santosh G (EE19B055)
PRML Assignment 3
'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import det_curve
from mpl_toolkits import mplot3d
from sklearn.metrics import DetCurveDisplay
import scipy.stats as scs

K = 25 # no of clusters

train_data = 'Synthetic_Data/21/train.txt'

file1= open(train_data)
train1=np.array([[float(lines.split(',')[0]),float(lines.split(',')[1]),int(lines.split(',')[2])] for lines in file1.readlines()],dtype=np.float64)
train_tb = pd.DataFrame(train1,columns=['x','y','class'])

class1 = train_tb.loc[train_tb['class']==1]
class2 = train_tb.loc[train_tb['class']==2]

dev_data = 'Synthetic_Data/21/dev.txt'
file1= open(dev_data)
dev1=np.array([[float(lines.split(',')[0]),float(lines.split(',')[1]),int(lines.split(',')[2])] for lines in file1.readlines()],dtype=np.float64)
dev_tb = pd.DataFrame(dev1,columns=['x','y','class'])				#we might need multi-variate not just bi-variate

def gaussian(x,mean,cov):
  return ((2*np.pi*np.sqrt(np.linalg.det(cov)))**(-1))*np.exp(-0.5*np.inner((x-mean),np.matmul(np.linalg.pinv(cov),(x-mean))))

np.random.seed(42)                                                      #K-Means Computation
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
    while np.linalg.norm(np.array(new_centers)-np.array(old_centers))>0.0001:
        #print(new_centers)
        old_centers = new_centers
        labels = pt_assign(old_centers,data)
        #print(labels)
        new_centers = update_centers(old_centers,data, labels, k)
        #print(new_centers)
     
    return  labels,new_centers

def mat_cal(data, K, pi, mean, cov):
  mat = np.zeros((len(data), K),dtype=np.float64)
  denom = np.zeros(len(data),dtype=np.float64)
 
  for k in range (K):
    for i in range(len(data)):
      mat[i][k] = pi[k]*gaussian(data[i],mean[k],cov[k])
   
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
  return mean,cov

def density_curve(mean,cov,params):
  def gaussian_graph(mean,cov,x,y):
    x=x-mean[0]
    y=y-mean[1]
    c=np.linalg.pinv(cov)
    return ((2*np.pi*np.sqrt(np.linalg.det(cov)))**(-1))*np.exp(-0.5*(c[0,0]*(x**2)+(c[1,0]+c[0,1])*x*y+c[1,1]*(y**2)))
  x=np.linspace(params[0][0],params[0][1],100)
  y=np.linspace(params[1][0],params[1][1],100)
  #Z = np.zeros(x.shape[0]*y.shape[0])
  X,Y = np.meshgrid(x,y)
 # data_points = np.c_[X.ravel(),Y.ravel()]
 # for i,p in enumerate(data_points):
 #   Z[i] = int(predictor(p))
  #Z = np.reshape(Z,X.shape)
  #plt.figure()
  for i in range(K):
    plt.contour(X, Y, gaussian_graph(mean[i], cov[i],X,Y),[0.005, 0.01, 0.03, 0.05,0.07, 0.1, 0.2, 0.3, 0.4])
  
  #plt.scatter(class1[['x','y']].values[:,0], class1[['x','y']].values[:,1], marker='^',facecolors='none', edgecolors='r', label='Class 1')
  plt.grid()

def predictor(x):
  if(np.max([gaussian(x, mean2[i], cov2[i]) for i in range(K)]) > np.max([gaussian(x, mean[i], cov[i]) for i in range(K)])):
    return 2

  if(np.max([gaussian(x, mean2[i], cov2[i]) for i in range(K)]) < np.max([gaussian(x, mean[i], cov[i]) for i in range(K)])):
    return 1


def decision_boundary(mean,cov,params,mnum):
  #x = np.arange(params[0][0],params[0][1],0.2)
  #y = np.arange(params[1][0],params[1][1],0.2)
  x=np.linspace(params[0][0],params[0][1],100)
  y=np.linspace(params[1][0],params[1][1],100)
  Z = np.zeros(x.shape[0]*y.shape[0])
  X,Y = np.meshgrid(x,y)
  data_points = np.c_[X.ravel(),Y.ravel()]
  for i,p in enumerate(data_points):
    Z[i] = int(predictor(p))
  Z = np.reshape(Z,X.shape)
  plt.figure()

  plt.title('Decision boundary of the model-')
  plt.contourf(X,Y,Z, extend='both', colors=('#FFFF00', '#ff9900'))
  density_curve(mean,cov,params)
  density_curve(mean2,cov2,params)
  plt.legend()
  plt.show()

def ROC(mean,cov,groundtruth=dev_tb[['class']].values):
  #S = np.zeros((len(dev_tb),3))
  #for i in range(len(dev_tb)):
  #  for j in range(3):
  S= [[np.max([gaussian(x, mean[i], cov[i]) for i in range(K)]),np.max([gaussian(x, mean2[i], cov2[i]) for i in range(K)])] for x in dev_tb[['x','y']].values ]

  thresh = np.linspace(np.amin(S),np.amax(S),200)
  TPR=[]
  FPR=[]
  FNR=[]
  for t in thresh:
    tp=0
    fp=0
    tn=0
    fn=0

    for i in range(len(dev_tb)):
      for j in range(2):
        if S[i][j]>=t:
          if groundtruth[i] == j+1:
            tp+=1
          else:
            fp+=1
        else:
          if groundtruth[i] == j+1:
            fn+=1
          else:
            tn+=1

    TPR.append(tp/(tp+fn))
    FPR.append(fp/(fp+tn))
    FNR.append(fn/(tp+fn))

  return FPR,TPR,FNR

train_data = 'Synthetic_Data/21/train.txt'
file1= open(train_data)
train1=np.array([[float(lines.split(',')[0]),float(lines.split(',')[1]),int(lines.split(',')[2])] for lines in file1.readlines()],dtype=np.float64)
train_tb = pd.DataFrame(train1,columns=['x','y','class'])

class1 = train_tb.loc[train_tb['class']==1]
class2 = train_tb.loc[train_tb['class']==2]

dev_data = 'Synthetic_Data/21/dev.txt'
file1= open(dev_data)
dev1=np.array([[float(lines.split(',')[0]),float(lines.split(',')[1]),int(lines.split(',')[2])] for lines in file1.readlines()],dtype=np.float64)
dev_tb = pd.DataFrame(dev1,columns=['x','y','class'])

class1['cluster'],centers = kmeans(K,class1[['x','y']].values)

plt.figure()
for i in range(K):
  int_df = class1.loc[class1['cluster']==i]                                      #please ignore a warning associated with this line
  plt.scatter(int_df[['x','y']].values[:,0], int_df[['x','y']].values[:,1])
  plt.plot(centers[i][0],centers[i][1],marker="x",markersize=20)
plt.show()

x_limits=(-20,20)
y_limits=(-20,20)

params=[x_limits,y_limits]

pi_i = np.full(K,1/K,dtype=np.float64)

mean_i = centers

cov_i = [np.array([[1,0],[0,1]]) for i in range(K)]

mean,cov = gmm(class1[['x','y']].values,mean_i,cov_i,pi_i,iters=6)

density_curve(mean,cov,params)
# plt.figure()
# for i in range(K):
#   int_df = class1.loc[class1['cluster']==i]
#   plt.scatter(int_df[['x','y']].values[:,0], int_df[['x','y']].values[:,1])
#   plt.plot(mean[i][0],mean[i][1],marker="x",markersize=20)
# plt.show()
labels,centers = kmeans(K,class2[['x','y']].values)
#kmeans.labels_

class2['cluster'] = labels

plt.figure()
for i in range(K):
  int_df = class2.loc[class2['cluster']==i]
  plt.scatter(int_df[['x','y']].values[:,0], int_df[['x','y']].values[:,1])
  plt.plot(centers[i][0],centers[i][1],marker="x",markersize=20)
plt.show()

#Initial estimates of pi_k, mu_k and sigma_k
pi_1 = np.full(K,1/K,dtype=np.float64)

mean_1 = centers

cov_1 = [np.array([[1,0],[0,1]]) for i in range(K)]

mean2,cov2 = gmm(class2[['x','y']].values,mean_1,cov_1,pi_1,iters=5)

density_curve(mean2,cov2,params)

pred_list = []

for x in dev_tb[['x','y']].values:
  pred_list.append(predictor(x))

confusion_matrix(dev_tb['class'],pred_list)

decision_boundary(mean,cov,params,mnum=1)
plt.figure()
plt.plot(ROC(mean,cov)[0],ROC(mean,cov)[1])
plt.show
fig = plt.figure(figsize=(9,6))
ax = fig.gca

plt.figure()
DetCurveDisplay(fpr = ROC(mean,cov)[0], fnr = ROC(mean,cov)[2], estimator_name = 'K = ' + str(K)).plot()
plt.title("DET-CURVE")
plt.show()

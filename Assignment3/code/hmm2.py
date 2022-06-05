
#HMM implemented for handwritten data

tm1 = ['1','3','5','9','z']
tm2 = ['a','ai','bA','chA','lA']

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


def recenter(points):
  for i in range(len(points)):
    x_centers = (np.amax(points[i][:,0])+np.amin(points[i][:,0]))/2
    y_centers = (np.amax(points[i][:,1])+np.amin(points[i][:,1]))/2
    center = np.vstack((x_centers, y_centers)).T
    points[i] = points[i] - center
    points[i][:,0] = points[i][:,0]/(np.amax(points[i][:,0])-np.amin(points[i][:,0]))
    points[i][:,1] = points[i][:,1]/(np.amax(points[i][:,1])-np.amin(points[i][:,1]))
  return points

def plot_points(points,arr):
  plt.figure(figsize=(20,20))
  for n in range(len(points)) : 
      plt.subplot(arr[0],arr[1],n+1)
      plt.plot(points[n][:,0],points[n][:,1]) 


template_list2 = []
template_list2.append(recenter(get_points('Handwriting_Data/a/train')))
template_list2.append(recenter(get_points('Handwriting_Data/ai/train')))
template_list2.append(recenter(get_points('Handwriting_Data/bA/train')))
template_list2.append(recenter(get_points('Handwriting_Data/chA/train')))
template_list2.append(recenter(get_points('Handwriting_Data/lA/train')))


dev_list2 = []
dev_label2 = []
for t in range(5):
  temp = recenter(get_points('Handwriting_Data/'+tm2[t]+'/dev'))
  dev_label2 = dev_label2 +[t for i in range(len(temp))]
  dev_list2 = dev_list2+temp



comp_list2 = []
for i in range(len(template_list2)):
  for j in range(len(template_list2[i])):
    comp_list2 += template_list2[i][j].tolist()




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
        labels.append(np.argmin([ np.linalg.norm(np.array(pt)-np.array(cen)) for cen in centers ]))
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
    i=0
    while (np.linalg.norm(np.array(new_centers)-np.array(old_centers))>0.05) and i<10:
        #print(new_centers)
        old_centers = new_centers
        labels = pt_assign(old_centers,data)
        #print(labels)
        new_centers = update_centers(old_centers,data, labels, k)
        #print(new_centers)
        print(np.linalg.norm(np.array(new_centers)-np.array(old_centers)))
        i+=1
    return  labels,new_centers


K2 = 60


print("Starting k-means")
labels2,centers2 = kmeans(K2,comp_list2)
print("Done...")


plt.scatter(np.array(comp_list2)[:,0],np.array(comp_list2)[:,1])
plt.scatter(np.array(centers2)[:,0],np.array(centers2)[:,1],marker="o",color='r')




symb2=[]
for i in range(len(template_list2)):
  temp1=[]
  for j in range(len(template_list2[i])):
    temp1.append(np.argmin(np.transpose([np.sum((template_list2[i][j] - cen)**2,axis=1) for cen in centers2]),axis=1))
  symb2.append(temp1)


dev_list2=[]
dev_label2 = []
for t in range(5):
  temp = recenter(get_points('Handwriting_Data/'+tm2[t]+'/dev'))
  dev_label2= dev_label2 +[t for i in range(len(temp))]
  dev_list2 = dev_list2+temp


symb_dev2=[]
for i in range(len(dev_list2)):
    symb_dev2.append(np.argmin(np.transpose([np.sum((dev_list2[i] - cen)**2,axis=1) for cen in centers2]),axis=1))



states = [10,10,11,11,10]
symbols = K2


original_dir = os.getcwd()
os.chdir(original_dir+'/HMM-Code')

params2=[]

for i in range(len(symb2)):
  with open('test.hmm.seq', 'w') as outfile:
    for j in range(len(symb2[i])):
      for k in range(len(symb2[i][j])):
        outfile.write(str(int(symb2[i][j][k]))+' ')
      outfile.write('\n')


  os.system(f"./train_hmm test.hmm.seq 1234 {states[i]} {symbols} .01")\


  fl = open('test.hmm.seq.hmm')
  lst=[]
  for ls in [line.split() for line in fl.readlines()[2:-2]]:
    if ls != []:
        lst.append(np.float_(ls))
  params2.append(lst)


os.chdir(original_dir)



tr_prob2=[]
sym_prob2 =[]
for lt in params2:
    Ap = []
    B = []
    for i in range(len(lt)):
        ind =int(i/2)
        Ap.append(lt[i][0])
        B.append(lt[i][1:])

    Ap[-1]=0
    Ap = np.reshape(Ap,(int(len(Ap)/2),2))

    Bp=[]
    for i in range(0,len(lt),2):
        Bp.append(B[i:i+2])
        
    tr_prob2.append(Ap)
    sym_prob2.append(Bp)


def prob_calc(sls,A,B):
    ns = len(A)

    mat = [[0 for i in range(ns)] for _ in range(len(sls))]
    
    for i in range(len(sls)):
        for j in range(ns):
            
            if j != 0: 
                t1 = A[j-1][1] * B[j-1][1][sls[i]]  #prob of transit from previous state
                
            t2 = A[j][0] * B[j][0][sls[i]]   #prob of transit from the same state
            
            if j != 0 and i != 0:
                mat[i][j] = t1*mat[i-1][j-1]+t2*mat[i-1][j]
                
            if j == 0 and i != 0:
                mat[i][j] = t2*mat[i-1][j]
                
            if i == 0:
                mat[i][j] = t2           
    
    prob = 0
    for j in range(ns) : 
        prob += mat[len(sls)-1][j]
    return prob

scores2=[]
pred_label2=[]
for smp in symb_dev2:
    scores2.append([prob_calc(smp,tr_prob2[i],sym_prob2[i]) for i in range(5)])
    pred_label2.append(np.argmax(scores2[-1]))




print(confusion_matrix(dev_label2,pred_label2))



def ROC(scores,dev_label):
  #S = np.zeros((len(dev_tb),3))
  #for i in range(len(dev_tb)):
  #  for j in range(3):

  thresh = np.sort(list(set(np.ravel(scores))))
  #thresh = np.linspace(np.min(scores),np.max(scores),200)  
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
  

norm_scores2 = [i/np.sum(i) for i in scores2]

plt.plot(ROC(norm_scores2,dev_label2)[0],ROC(norm_scores2,dev_label2)[1],color='k')
#lt.legend()
plt.grid()
plt.ylabel('TPR')
plt.xlabel('FPR')
plt.title("ROC-CURVE")

m1 = ROC(norm_scores2,dev_label2)
fig = plt.figure(figsize=(7,5))
ax= fig.gca()
DetCurveDisplay(fpr=m1[0],fnr=m1[2]).plot(ax)

plt.show()






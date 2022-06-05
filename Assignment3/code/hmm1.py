#HMM implemented for isolated digits




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


template_list=[]
template_list.append(get_template('Isolated_Digits/'+tm1[0]+'/train'))
template_list.append(get_template('Isolated_Digits/'+tm1[1]+'/train'))
template_list.append(get_template('Isolated_Digits/'+tm1[2]+'/train'))
template_list.append(get_template('Isolated_Digits/'+tm1[3]+'/train'))
template_list.append(get_template('Isolated_Digits/'+tm1[4]+'/train'))

print("Doing k-means")
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
    for i in range(12):
        #print(new_centers)
        old_centers = new_centers
        labels = pt_assign(old_centers,data)
        #print(labels)
        new_centers = update_centers(old_centers,data, labels, k)
        #print(new_centers)
        print(np.linalg.norm(np.array(new_centers)-np.array(old_centers)))
    return  labels,new_centers



comp_list = []
for i in range(len(template_list)):
  for j in range(len(template_list[i])):
    comp_list += template_list[i][j]



K = 25 # no of clusters


labels,centers = kmeans(K,comp_list)

print("Done...")


symb=[]
for i in range(len(template_list)):
  temp1=[]
  for j in range(len(template_list[i])):
    temp1.append(np.argmin(np.transpose([np.sum((template_list[i][j] - cen)**2,axis=1) for cen in centers]),axis=1))
  symb.append(temp1)
      


dev_list=[]
dev_label = []
for t in range(5):
  temp = get_template('Isolated_Digits/'+tm1[t]+'/dev')
  dev_label= dev_label +[t for i in range(len(temp))]
  dev_list = dev_list+temp


symb_dev=[]
for i in range(len(dev_list)):
    symb_dev.append(np.argmin(np.transpose([np.sum((dev_list[i] - cen)**2,axis=1) for cen in centers]),axis=1))



states = [5,5,5,5,6]
symbols = K


original_dir = os.getcwd()
os.chdir(original_dir+'/HMM-Code')

params=[]

for i in range(len(symb)):
  with open('test.hmm.seq', 'w') as outfile:
    for j in range(len(symb[i])):
      for k in range(len(symb[i][j])):
        outfile.write(str(int(symb[i][j][k]))+' ')
      outfile.write('\n')


  os.system(f"./train_hmm test.hmm.seq 1234 {states[i]} {symbols} .01")\


  fl = open('test.hmm.seq.hmm')
  lst=[]
  for ls in [line.split() for line in fl.readlines()[2:-2]]:
    if ls != []:
        lst.append(np.float_(ls))
  params.append(lst)

os.chdir(original_dir)



tr_prob=[]
sym_prob =[]
for lt in params:
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
        
    tr_prob.append(Ap)
    sym_prob.append(Bp)



#function for testing


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





pred_label=[]
scores = []
for smp in symb_dev:
    scores.append([prob_calc(smp,tr_prob[i],sym_prob[i]) for i in range(5)])
    pred_label.append(np.argmax(scores[-1]))


print(confusion_matrix(dev_label,pred_label))



def ROC(scores,dev_label):
  #S = np.zeros((len(dev_tb),3))
  #for i in range(len(dev_tb)):
  #  for j in range(3):
  thresh = np.sort(list(set(np.ravel(scores))))
  #thresh = np.linspace(np.amin(scores),np.amax(scores),200)
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

norm_scores = [i/np.sum(i) for i in scores]

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
















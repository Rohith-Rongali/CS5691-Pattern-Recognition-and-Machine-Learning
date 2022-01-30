import numpy as np
import matplotlib.pyplot as plt
#import cv2

print("Total of 5 figures")
print("Fig1 loading.....")
test_img = plt.imread('76.jpg',0)
test_img = np.array(test_img,dtype = np.float64)
plt.imshow(test_img,cmap='gray',vmin=0,vmax=255)
plt.title("Fig1:Original image",size='x-large')

print("Fig2 loading.....")
def evd_rct(test_img,k=100):                            #returns the image reconstructed using evd(default k=100)
  eigval,eigvec = np.linalg.eig(test_img)
  order = np.absolute(eigval).argsort()[::-1]
  eigval = eigval[order]
  eigvec = eigvec[:,order]
  img_rct = np.dot(np.dot(eigvec[:,:k],np.diag(eigval[:k])),np.linalg.inv(eigvec)[:k,:])
  return [eigval,eigvec,img_rct]


#Eigenvalue plot
eigvals = evd_rct(test_img)[0]
plt.figure(figsize=(9,6))
plt.plot(np.absolute(eigvals))
plt.title("Fig2: Plot of eigenvalues magnitude(descending order) in log-scale",size='x-large')
plt.yscale('log')
plt.xlabel("k")
plt.ylabel("eigenvalues magnitude")
plt.grid()


#finding those of k that include all conjugate pairs of eigenvalues
print("Fig3(2-parts) loading.....")
x1=[]
x2=[]
y1=[]
y2=[]
eigvals=evd_rct(test_img,256)[0]
for i in range(1,257):
  if np.absolute(np.imag(np.sum(eigvals[:i]))) < 1e-10:
    x1.append(i)             #benign index, all conjugate pairs included		
  else:
    x2.append(i)




#The below code checks if the imaginary part of reconstructed matrix(indices from x1) is zero(<1e-10). I commented it out since it slightly slows down the program
'''
#plotting % of pixels having imaginary part nearly zero(checking the imaginary part of reconstructed images with benign values of k)
print("Running the optional check")
for i in range(1,257):
  if np.absolute(np.imag(np.sum(eigvals[:i]))) < 1e-10:                                  						
    y1.append(np.count_nonzero(np.imag(evd_rct(test_img,i)[2])<=1e-10)/(256*2.56)) 		
  else:
    y2.append(np.count_nonzero(np.imag(evd_rct(test_img,i)[2])<=1e-10)/(256*2.56))

plt.figure(figsize=(8,12))
plt.subplot(211)
plt.plot(x1,y1,marker='o',color='green')
plt.title("Fig3: % of pixels having imaginary part almost zero in reconstructed images(including conjugate pairs)",size='x-large')
plt.ylabel("percentage of pixels")
plt.grid()
plt.subplot(212)
plt.plot(x2,y2,marker='v',color='red')
plt.xlabel("k")
plt.ylabel("percentage of pixels")
plt.title("% of pixels having imaginary part almost zero in reconstructed images(not including conjugate pairs)",size='x-large')
plt.grid()
print("Completed optional check.")
'''



from matplotlib import axes
#x1 contains all the benign k-values that we can consider
k_list1=[10,25,50] #sample values to plot
k_list2=[99,150,256]

def plot_evd(k_list):
  fig,axarr = plt.subplots(nrows=3,ncols=2,figsize=(5,10))
  i=0
  for k in k_list:
    img_rct1 = evd_rct(test_img,k)[2]
    axarr[i,0].set(xlabel="Error = "+str(np.linalg.norm(np.real(img_rct1)-test_img,'fro')),ylabel="k value = "+str(k))
    axarr[i,0].imshow(np.real(img_rct1),cmap='gray',vmin=0,vmax=255)
    if i==0:
      axarr[0,0].set_title("Reconstructed plot")
    axarr[i,1].imshow(np.absolute(test_img-np.real(img_rct1)),cmap='gray',vmin=0,vmax=255)
    if i==0:
      axarr[0,1].set_title("diff from original")
    i=i+1
  fig.suptitle("Fig3:Reconstructed images for values of k ="+str(k_list),size='xx-large')
  #fig.tight_layout()
 

plot_evd(k_list1)
plot_evd(k_list2)

print("Fig4 loading.....")
fnorm_evd=[]
for x in x1:
  fnorm_evd.append(np.linalg.norm(np.real(evd_rct(test_img,x)[2])-test_img,'fro'))


plt.figure(figsize=(9,6))
plt.plot(x1,fnorm_evd,marker='o',label='evd')
plt.xlabel("k")
plt.ylabel("Error")
plt.title("Fig4:Error plot for EVD reconstructions",size='x-large')
plt.legend()
plt.grid()
plt.show()
print("Done")




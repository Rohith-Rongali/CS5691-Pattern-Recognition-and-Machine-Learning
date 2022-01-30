

import numpy as np
import matplotlib.pyplot as plt


test_img = plt.imread('76.jpg',0)
test_img = np.array(test_img,dtype = np.float64)
plt.imshow(test_img,cmap='gray')
plt.title("Fig1:Original image",size='x-large')


def img_rct_svd(A,k=10):
  A_t= np.transpose(A)
  left_m = np.dot(A,A_t)
  right_m=np.dot(A_t,A)
  sing_vals = np.sqrt(np.linalg.eigh(left_m)[0][::-1],dtype = np.float64)
  right_eig_vec = np.transpose(np.linalg.eigh(right_m)[1][:,::-1])		
  sing_vals_inv = sing_vals.copy()
  sing_vals_inv[np.where(sing_vals!=0)] = 1/sing_vals_inv[np.where(sing_vals!=0)]			
  left_eig_vec = np.dot(A,np.dot(np.transpose(right_eig_vec),np.diag(sing_vals_inv)))
  rst_img = np.dot(left_eig_vec[:,:k],np.diag(sing_vals[:k]))
  rst_img = np.dot(rst_img, right_eig_vec[:k,:])
  return [left_eig_vec,sing_vals,right_eig_vec,rst_img]
  
singvals = img_rct_svd(test_img)[1]
plt.figure(figsize=(9,6))
plt.plot(np.absolute(singvals))
plt.yscale('log')
plt.xlabel("k")
plt.ylabel("singular-values magnitude")
plt.grid()
plt.title("Fig2: Plot of singular values(descending order) in log-scale",size='x-large')


from matplotlib import axes

k_list1=[10,25,50] #sample values to plot
k_list2=[100,150,256]

def plot_svd(k_list):
  fig,axarr = plt.subplots(nrows=3,ncols=2,figsize=(8,15))
  i=0
  for k in k_list:
    img_rct1 = img_rct_svd(test_img,k)[3]
    axarr[i,0].imshow(np.real(img_rct1),cmap='gray',vmin=0,vmax=255)
    axarr[i,0].set(xlabel="Error = "+str(np.linalg.norm(np.real(img_rct1)-test_img,'fro')),ylabel="k value = "+str(k))
    if i==0:
      axarr[0,0].set_title("Reconstructed plot")
    axarr[i,1].imshow(test_img-np.real(img_rct1),cmap='gray',vmin=0,vmax=255)
    if i==0:
      axarr[0,1].set_title("diff from original")
    i=i+1
  fig.suptitle("Fig3:Reconstructed images for some values of k ="+str(k_list),size='xx-large')
  #fig.tight_layout()


plot_svd(k_list1)
plot_svd(k_list2)




fnorm = []
for x in range(1,257):
  fnorm.append(np.linalg.norm(img_rct_svd(test_img,x)[3]-test_img,'fro'))

plt.figure(figsize=(9,6))
plt.plot(range(1,257),fnorm,marker='o')
plt.grid()
plt.xlabel("k")
plt.ylabel("Error")
plt.title("Fig4: Error plot for SVD reconstructions",size='x-large')



#for color image


color_img = plt.imread('lena.png')
color_img = np.array(color_img,dtype = np.float64)
plt.figure(figsize=(9,6))
plt.imshow(color_img)
plt.title("Fig5: Original Color Image",size='x-large')


R = color_img[:,:,0]   #perform invidually for each color channel
G = color_img[:,:,1]
B = color_img[:,:,2]
zeros = np.zeros(color_img.shape)
Norm = np.linalg.norm(R,'fro')+np.linalg.norm(G,'fro')+np.linalg.norm(B,'fro')

k_list = [10,40,70]
img = np.zeros(color_img.shape)
fig,axarr = plt.subplots(3,2,figsize=(7,12))
i=0
for k in k_list:
  img[:,:,0] = img_rct_svd(R,k)[3]
  img[:,:,1] = img_rct_svd(G,k)[3]
  img[:,:,2] = img_rct_svd(B,k)[3]
  axarr[i,0].imshow(img)
  axarr[i,0].set(xlabel="Error = "+str(np.linalg.norm(R-img[:,:,0],'fro')+np.linalg.norm(G-img[:,:,1],'fro')+np.linalg.norm(B-img[:,:,2],'fro')),ylabel='k = '+str(k))
  axarr[i,1].imshow(color_img-img)
  axarr[i,1].set(xlabel="diff from original")
  i=i+1
fig.suptitle("Fig6: Reconstructed color images",size='xx-large')


plt.show()

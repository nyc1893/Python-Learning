# -*- coding: utf-8 -*-

"""
Created on Mon Mar 15 13:27:26 2021

@author: 006501205
"""

# %matplotlib inline
import matplotlib.pyplot as plt 
import os
import numpy as np
from scipy.io import loadmat
from matrix_completion import *
from scipy.linalg import svd
import time
import six
import sys
sys.modules['sklearn.externals.six'] = six
# import mlrose
from fancyimpute import (
    BiScaler,
    KNN,
#    Nuclea1rNormMinimization,
    SoftImpute,
    SimpleFill
)

# 
x= np.load('2016_Jan_07_4_Frequency Event_tsg.npy')
x=x[:,:,10000:25000]
plt.plot(x[0,0,:])
plt.show()

# X_org=tensor.Tensor(x)
X =  x
# Second: create Tensor object and find missing data
#X0 = X.data.copy()
X0 = x.copy()
ind_nan=np.argwhere(np.isnan(x))
t_window=60
old_window=60
time_steps=int(int(x.shape[2])/t_window)
ten = []
omega = []
sol = []
batch_size=[int(x.shape[0]),int(x.shape[1]),t_window]
siz = np.array([batch_size for t in range(time_steps)])
ra=5
r = [ra, ra, ra]
dims = len(siz)
miss=0.2
# remove nan values
inds=np.where(np.isnan(x))
col_mean=[]
x[inds]=0
# inserting nan value with interopolation of 5 closest values
for ii in range(len(inds[0])):
   col_mean.append(np.mean(x[inds[0][ii],inds[1][ii],inds[2][ii]-5:inds[2][ii]+5]))
   x[inds[0][ii],inds[1][ii],inds[2][ii]]=col_mean[ii]
   
np.random.seed(0)   
for t in range(time_steps):
            temp_sol = x[:,:,t*t_window:(t+1)*t_window]
            temp_omega = (np.random.random(siz[t]) > miss) * 1
            temp_ten = temp_sol.copy()
            sol.append(temp_sol)
            temp_ten[temp_omega == 0] -= temp_ten[temp_omega == 0]
            omega.append(temp_omega)
            ten.append(temp_ten)

X1=ten
Omega1=omega
sol1=sol
errList_knn=[]
errList_softimpute=[]
errList_softImpute_no_biscale=[]
errList_SVT=[]
errList_BMF=[]
errList_PMF=[]
time_knn=[]
time_softimpute=[]
time_softimpute_no_biscale=[]
time_SVT=[]
time_BMF=[]
time_PMF=[]







lstnan = np.isnan(x)
x = np.nan_to_num(x)
Mat_Comp={}

for t in range(time_steps):
    
    print('Step '+ str(t)+'\n')

    missing_mask=Omega1[t].reshape([X1[t].shape[0]*X1[t].shape[1],X1[t].shape[2]])
    mask = missing_mask
    missing_mask=1-missing_mask
    missing_mask=missing_mask.astype(bool)

# missing_mask = np.random.rand(*X.shape) < 0.5
    X_incomplete = X1[t].reshape([X1[t].shape[0]*X1[t].shape[1],X1[t].shape[2]]).copy()
    # missing entries indicated with NaN
    X_incomplete[missing_mask] = np.nan

    X_nn= X1[t].reshape([X1[t].shape[0]*X1[t].shape[1],X1[t].shape[2]]).copy()
    X_nn[missing_mask]=0


# Use 3 nearest rows which have a feature to fill in each row's missing features
    knnImpute = KNN(k=3)
    start_knn = time.time()
    X_filled_knn = knnImpute.fit_transform(X_incomplete)
    elapsed_knn = time.time() - start_knn

# Instead of solving the nuclear norm objective directly, instead
# induce sparsity using singular value thresholding
    softImpute = SoftImpute()

# simultaneously normalizes the rows and columns of your observed data,
# sometimes useful for low-rank imputation methods
    biscaler = BiScaler()

# rescale both rows and columns to have zero mean and unit variance
    start_softImpute = time.time()
    X_incomplete_normalized = biscaler.fit_transform(X_incomplete)
    X_filled_softimpute_normalized = softImpute.fit_transform(X_incomplete_normalized)
    X_filled_softimpute = biscaler.inverse_transform(X_filled_softimpute_normalized)
    elapsed_softimpute = time.time() - start_softImpute
    
    start_softImpute_no_biscale = time.time()
    X_filled_softimpute_no_biscale = softImpute.fit_transform(X_incomplete)
    elapsed_softimpute_no_biscale = time.time() - start_softImpute_no_biscale

    start_SVT = time.time()
    X_filled_SVT = svt_solve(X_nn, mask)
    elapsed_SVT = time.time() - start_SVT
    
    start_PMF = time.time()
    X_filled_pmf = pmf_solve(X_nn, mask, 10 , 1e-2)
    elapsed_PMF = time.time() - start_PMF
    
    start_BMF = time.time()
    X_filled_bmf = biased_mf_solve(X_nn, mask, 10, 1e-2)
    elapsed_BMF = time.time() - start_BMF

    time_knn.append(elapsed_knn)
    time_softimpute.append(elapsed_softimpute)
    time_softimpute_no_biscale.append(elapsed_softimpute_no_biscale)
    time_SVT.append(elapsed_SVT)
    time_BMF.append(elapsed_BMF)
    time_PMF.append(elapsed_PMF)

    realX = sol1[t].reshape([X1[t].shape[0]*X1[t].shape[1],X1[t].shape[2]])
    norm1 = np.linalg.norm(realX)
    norm2 = np.linalg.norm(realX * (1-mask))
    
# print relative errors for the imputation methods above
    
    softImpute_err1 = np.linalg.norm(X_filled_softimpute - realX)  # Absolute Error
    softImpute_err2 = np.linalg.norm((X_filled_softimpute - realX) * (1 - mask))
    softImpute_re_err1 = softImpute_err1 / norm1  # Relative Error 1
    softImpute_re_err2 = softImpute_err2 / norm2  # Relative Error 2
    # meanfill_mse = ((X_filled_mean[missing_mask] - X[missing_mask]) ** 2).mean()
    # print("The Relative Error of the softImpute are: %f, %f, %f" % (softImpute_err1, softImpute_re_err1, softImpute_re_err2))
    # Mat_Comp.update({(miss,'softImpute'):(softImpute_err1, softImpute_re_err1, softImpute_re_err2)})
    errList_softimpute.append(softImpute_re_err1)

    softImpute_no_biscale_err1 = np.linalg.norm(X_filled_softimpute_no_biscale - realX)  # Absolute Error
    softImpute_no_biscale_err2 = np.linalg.norm((X_filled_softimpute_no_biscale - realX) * (1 - mask))
    softImpute_no_biscale_re_err1 = softImpute_no_biscale_err1 / norm1  # Relative Error 1
    softImpute_no_biscale_re_err2 = softImpute_no_biscale_err2 / norm2  # Relative Error 2
    # print("The Relative Error of the softImpute_no_biscale are: %f, %f, %f" % (softImpute_no_biscale_err1,softImpute_no_biscale_re_err1,softImpute_no_biscale_re_err2))
    # Mat_Comp.update({(miss,'softImpute_no_biscale'):(softImpute_no_biscale_err1,softImpute_no_biscale_re_err1,softImpute_no_biscale_re_err2)})
    errList_softImpute_no_biscale.append(softImpute_no_biscale_re_err1)

    knnImpute_err1 = np.linalg.norm(X_filled_knn - realX)  # Absolute Error
    knnImpute_err2 = np.linalg.norm((X_filled_knn - realX) * (1 - mask))
    knnImpute_re_err1 = knnImpute_err1 / norm1  # Relative Error 1
    knnImpute_re_err2 = knnImpute_err2 / norm2  # Relative Error 2
    # print("The Relative Error of the knnImpute are: %f, %f, %f" % (knnImpute_err1,knnImpute_re_err1,knnImpute_re_err2))
    # Mat_Comp.update({(miss,'KNN'):(knnImpute_err1,knnImpute_re_err1,knnImpute_re_err2)})
    errList_knn.append(knnImpute_re_err1)

    SVT_err1 = np.linalg.norm(X_filled_SVT - realX)  # Absolute Error
    SVT_err2 = np.linalg.norm((X_filled_SVT - realX) * (1 - mask))
    SVT_re_err1 = SVT_err1 / norm1  # Relative Error 1
    SVT_re_err2 = SVT_err2 / norm2  # Relative Error 2
    # print("The Relative Error of the SVT are: %f, %f, %f" % (SVT_err1,SVT_re_err1,SVT_re_err2))
    # Mat_Comp.update({(miss,'SVT'):(SVT_err1,SVT_re_err1,SVT_re_err2)})
    errList_SVT.append(SVT_re_err1)

    pmf_err1 = np.linalg.norm(X_filled_pmf - realX)  # Absolute Error
    pmf_err2 = np.linalg.norm((X_filled_pmf - realX) * (1 - mask))
    pmf_re_err1 = pmf_err1 / norm1  # Relative Error 1
    pmf_re_err2 = pmf_err2 / norm2  # Relative Error 2
    # print("The Relative Error of the PMF are: %f, %f, %f" % (pmf_err1,pmf_re_err1,pmf_re_err2))
    # Mat_Comp.update({(miss,'PMF'):(pmf_err1,pmf_re_err1,pmf_re_err2)})
    errList_PMF.append(pmf_re_err1)

    bmf_err1 = np.linalg.norm(X_filled_bmf - realX)  # Absolute Error
    bmf_err2 = np.linalg.norm((X_filled_bmf - realX) * (1 - mask))
    bmf_re_err1 = bmf_err1 / norm1  # Relative Error 1
    bmf_re_err2 = bmf_err2 / norm2  # Relative Error 2
    # print("The Relative Error of the BMF are: %f, %f, %f" % (bmf_err1,bmf_re_err1,bmf_re_err2))
    # Mat_Comp.update({(miss,'BMF'):(bmf_err1,bmf_re_err1,bmf_re_err2)})   
    errList_BMF.append(bmf_re_err1)


fig = plt.figure(figsize=(30,16))
plt.subplot(6, 2, 1)       
plt.plot(time_knn)
plt.title('time elapsed at each time step for knn, mean is '+ str(round(np.mean(time_knn),3)))
# plt.show()

plt.subplot(6, 2, 2)
plt.title('RRE at each time step for knn, mean is '+ str(round(np.mean(errList_knn),3)))
plt.plot(errList_knn)
#plt.show()

plt.subplot(6, 2, 3)
plt.plot(time_softimpute)
plt.title('time elapsed at each time step for softimpute, mean is '+ str(round(np.mean(time_softimpute),3)))
#plt.show()

plt.subplot(6, 2, 4)
plt.plot(errList_softimpute)
plt.title('RRE at each time step for softimpute, mean is '+ str(round(np.mean(errList_softimpute),3)))
#plt.show()


plt.subplot(6, 2, 5)
plt.plot(time_softimpute_no_biscale)
plt.title('time elapsed at each time step for softimpute_no_biscale, mean is '+ str(round(np.mean(time_softimpute_no_biscale),3)))
#plt.show()

plt.subplot(6, 2, 6)
plt.plot(errList_softImpute_no_biscale)
plt.title('RRE at each time step for softimpute_no_biscale, mean is '+ str(round(np.mean(errList_softImpute_no_biscale),3)))
#plt.show()

plt.subplot(6, 2, 7)
plt.plot(time_SVT)
plt.title('time elapsed at each time step for SVT, mean is '+ str(round(np.mean(time_SVT),3)))
#plt.show()

plt.subplot(6, 2, 8)
plt.plot(errList_SVT)
plt.title('RRE at each time step for SVT, mean is '+ str(round(np.mean(errList_SVT),3)))
plt.show()

plt.subplot(6, 2, 9)
plt.plot(time_BMF)
plt.title('time elapsed at each time step for BMF, mean is '+ str(round(np.mean(time_BMF),3)))
#plt.show()

plt.subplot(6, 2, 10)
plt.plot(errList_BMF)
plt.title('RRE at each time step for BMF, mean is '+ str(round(np.mean(errList_BMF),3)))
plt.show()

plt.subplot(6, 2, 11)
plt.plot(time_PMF)
plt.title('time elapsed at each time step for PMF, mean is '+ str(round(np.mean(time_PMF),3)))
#plt.show()

plt.subplot(6, 2, 12)
plt.plot(errList_PMF)
plt.title('RRE at each time step for PMF, mean is '+ str(round(np.mean(errList_PMF),3)))
plt.show()
    


print('time elapsed at each time step for knn, mean is '+ str(round(np.mean(time_knn),3)))
print('time elapsed at each time step for softimpute, mean is '+ str(round(np.mean(time_softimpute),3)))
print('time elapsed at each time step for softimpute_no_biscale, mean is '+ str(round(np.mean(time_softimpute_no_biscale),3)))
print('time elapsed at each time step for SVT, mean is '+ str(round(np.mean(time_SVT),3)))
print('time elapsed at each time step for BMF, mean is '+ str(round(np.mean(time_BMF),3)))
print('time elapsed at each time step for PMF, mean is '+ str(round(np.mean(time_PMF),3)))
print('==========================================================================')
print('RRE at each time step for knn, mean is '+ str(round(np.mean(errList_knn),3)))
print('RRE at each time step for softimpute, mean is '+ str(round(np.mean(errList_softimpute),3)))
print('RRE at each time step for softimpute_no_biscale, mean is '+ str(round(np.mean(errList_softImpute_no_biscale),3)))
print('RRE at each time step for SVT, mean is '+ str(round(np.mean(errList_SVT),3)))
print('RRE at each time step for BMF, mean is '+ str(round(np.mean(errList_BMF),3)))
print('RRE at each time step for PMF, mean is '+ str(round(np.mean(errList_PMF),3)))
#plt.plot(X_filled_softimpute[0])
#plt.show()

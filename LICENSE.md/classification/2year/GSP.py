# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 08:52:00 2021
GSP sorting from this work:
Power System Event Identification based on Deep
Neural Network with Information Loading
@author: nyc
"""


import numpy as np
from scipy.stats import pearsonr

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
def pearsonrSim(x,y):
    '''
    皮尔森相似度
    '''
    return pearsonr(x,y)[0]

def gen_random():
    # 生成 10x10 的 数组
    st1 = []
    for i in range(10):     
        st1.append(np.squeeze(np.random.rand(10)))
        
    st1 = np.array(st1)
    path2 ="data/"
    np.save(path2+'pmu.npy',st1) 


path2 ="data/"
# pmu = np.load(path2+'pmu.npy') 
pmu = np.load(path2+'23-rof.npy') 
# print(st1.shape)
# print(pmu)
N= 23
w = np.zeros((N,N))
D = np.zeros((N,N))
for j in range(0,N):
    for i in range(0,N):
        w[j,i] = pearsonrSim(pmu[j],pmu[i])
        if(i==j):
            w[i,j]=0
            
            
for j in range(0,N):
    sum =0
    for i in range(0,N):
        sum += w[j,i]
    D[j,j] = sum
    
L = D - w
# print(L)
print(w.shape)
a, b = np.linalg.eig(L)
import heapq
# print(a)
x = heapq.nsmallest(2,a)
# print(x[1])
ind = np.where(a == x[1])[0][0]

# print(b[ind])
new_ind = np.argsort(b[ind])
print(new_ind)


X = np.arange(0,23)
Y = np.arange(0,23)
X, Y = np.meshgrid(X, Y)
plt.pcolor(X,Y, w, cmap='jet')
plt.colorbar()
# plt.ylabel("0.05-5Hz")
# plt.xlabel("Time(80s)")
plt.show()


pmu2 = np.zeros((pmu.shape[0],pmu.shape[1]))
for i in range(0,23):
    pmu2[i] = pmu[new_ind[i]]
w2 = np.zeros((N,N))
for j in range(0,N):
    for i in range(0,N):
        w[j,i] = pearsonrSim(pmu2[j],pmu2[i])
        if(i==j):
            w2[i,j]=0
            
            
plt.pcolor(X,Y, w, cmap='jet')
plt.colorbar()
# plt.ylabel("0.05-5Hz")
# plt.xlabel("Time(80s)")
plt.show()

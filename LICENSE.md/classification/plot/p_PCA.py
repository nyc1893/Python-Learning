
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pywt
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

# from plot_event import  removePlanned
from plot_event import read_data
from sklearn import decomposition
from sklearn import datasets

np.random.seed(5)


    
def Cwt(X_train):    
    wavename = 'morl'
    s_num = 4
    num = 3
    pca_2 = PCA(n_components=num)

    scales = np.arange(1, s_num+1)

    wavelet = []
    for i in range(0,X_train.shape[0]):
        data = X_train[i].flatten()
        [coeff1, freqs1] = pywt.cwt(data, scales, wavename)
        wavelet.append(pca_2.fit_transform(coeff1))
    wavelet = np.array(wavelet)
    print(wavelet.shape)
    return wavelet
    
# num = 4
# wavename = 'morl' 
# scales = np.arange(1, num+1)    
# i = 20
# data = X_test[i].flatten()
# [coeff1, freqs1] = pywt.cwt(data, scales, wavename)
# pca_2 = PCA(n_components=3)
# p2= pca_2.fit_transform(coeff1)

X_test, y_test = read_data()
print(X_test.shape)
# X_test = X_test[:,1,:,:]
X_test=  np.squeeze(X_test)
print(X_test.shape)
# X_test =  X_test[:4]
# y_test = y_test [:4]

p2 = Cwt(X_test)
print(p2.shape)
# print(p2)

# print(p2[:,:,1])
print(y_test.shape)
# print(y_test)


ind0 = np.where(y_test==0)[0]
ind1 = np.where(y_test==1)[0]
ind2 = np.where(y_test==2)[0]
ind3 = np.where(y_test==3)[0]

print(len(ind0))
print(len(ind1))
print(len(ind2))
print(len(ind3))


# print(p2[ind1,1,:])


# centers = [[1, 1], [-1, -1], [1, -1]]
iris = datasets.load_iris()
X = iris.data
y = iris.target
# print(type(X))
# print(X.shape)

# print(X)
# print(y)

fig = plt.figure(1, figsize=(4, 3))
plt.clf()
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

plt.cla()
pca = decomposition.PCA(n_components=3)
pca.fit(X)
X = pca.transform(X)
print(X.shape)
print(X[:, 1].shape)
print(np.unique(y))
y = np.choose(y, [1, 2, 0]).astype(np.float)

# scatter(x2, y2, z2, c='g', label='随机点')



ax.scatter(p2[ind0,1,0],p2[ind0,1,1],p2[ind0,1,2], c='g',label='Line')
ax.scatter(p2[ind1,1,0],p2[ind1,1,1],p2[ind1,1,2], c='r',label='Trans')
ax.scatter(p2[ind2,1,0],p2[ind2,1,1],p2[ind2,1,2], c='b',label='Fequency')
ax.scatter(p2[ind3,1,0],p2[ind3,1,1],p2[ind3,1,2], c='k',label='Oscilation')

ax.legend(loc='best')
ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])

plt.show()

"""
# centers = [[1, 1], [-1, -1], [1, -1]]
iris = datasets.load_iris()
X = iris.data
y = iris.target
# print(type(X))
# print(X.shape)

# print(X)
# print(y)

fig = plt.figure(1, figsize=(4, 3))
plt.clf()
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

plt.cla()
pca = decomposition.PCA(n_components=3)
pca.fit(X)
X = pca.transform(X)
print(X.shape)
print(X[:, 1].shape)
print(np.unique(y))
y = np.choose(y, [1, 2, 0]).astype(np.float)
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap=plt.cm.nipy_spectral,
           edgecolor='k')

# ax.w_xaxis.set_ticklabels([])
# ax.w_yaxis.set_ticklabels([])
# ax.w_zaxis.set_ticklabels([])

plt.show()
"""
"""

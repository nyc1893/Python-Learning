#get the sum and its lag out
#also concate the ws and deta wind direction which can be feed for already been trained Neat Genes. 

import pandas as pd
import numpy as np


df1 = pd.read_csv("C:/360Downloads/data/correct/remove-nan/mit_2009.csv")    
df2 = pd.read_csv("C:/360Downloads/data/correct/remove-nan/mit_2010.csv")    


df3 = pd.read_csv("C:/work/UNR/cs776/finial_project/1h/ppmit12_2009.csv")
df4 = pd.read_csv("C:/work/UNR/cs776/finial_project/1h/ppmit12_2010.csv")



lags = 8
# print(df.shape)
# print(df1.shape)
ind1 = df1.pop('ind')
ind2 = df2.pop('ind')


df3 = df3.iloc[ind1,[0,1]]
df3.columns = ['ws', 'detadir']
# print(df3.head(10))
df3 = df3.iloc[lags:,:]
print(df3.shape)


df4 = df4.iloc[ind2,[0,1]]
df4.columns = ['ws', 'detadir']
# print(df3.head(10))
df4 = df4.iloc[lags:,:]
print(df4.shape)

df1[df1<0]=0
df1['ColSum']=df1.apply(lambda x:x.sum()/1000,axis=1)
df2[df2<0]=0
df2['ColSum']=df2.apply(lambda x:x.sum()/1000,axis=1)
# print(df1.head())

train, test = [], []
attr = 'ColSum'
df1 = df1[attr].values
df2 = df2[attr].values

for i in range(lags, len(df1)):
	train.append(df1[i - lags: i + 1])
for i in range(lags, len(df2)):
	test.append(df2[i - lags: i + 1])	
train = np.array(train)
test = np.array(test)
# test = np.array(test)
# np.random.shuffle(train)

X_train = pd.DataFrame(train[:, :-1])
X_test = pd.DataFrame(test[:, :-1])
print(X_train.shape)
print(X_test.shape)
X_train.columns = ['l7','l6','l5','l4','l3','l2','l1','l0']
X_test.columns = ['l7','l6','l5','l4','l3','l2','l1','l0']
# X_train.reset_index()
# df3.reset_index()
X_train=pd.concat([df3,X_train],axis=1,join='inner')
X_test=pd.concat([df4,X_test],axis=1,join='inner')

print(X_train.shape)
print(X_test.shape)


print(X_train.head())
print(X_test.head())

X_train.to_csv("ppmit12_2009.csv",index =None)
X_test.to_csv("ppmit12_2010.csv",index =None)
print('mit data save done.')

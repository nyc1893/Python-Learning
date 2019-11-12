
#get the sum and its lag out
import pandas as pd
import numpy as np


df1 = pd.read_csv("C:/360Downloads/data/correct/remove-nan/mit_2009.csv")    
df2 = pd.read_csv("C:/360Downloads/data/correct/remove-nan/mit_2010.csv")    

# print(df1.head())
print(df1.shape)
df1[df1<0]=0
df1['ColSum']=df1.apply(lambda x:x.sum()/1000,axis=1)
# print(df1.head())

train, test = [], []
attr = 'ColSum'
df1 = df1[attr].values
lags = 6
for i in range(lags, len(df1)):
	train.append(df1[i - lags: i + 1])
	
train = np.array(train)
# test = np.array(test)
# np.random.shuffle(train)

X_train = pd.DataFrame(train[:, :-1])


print(type(X_train))
print(X_train.shape)


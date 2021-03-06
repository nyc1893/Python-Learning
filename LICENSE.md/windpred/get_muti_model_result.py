# This will help you to get multi model testing result

import pandas as pd
import numpy as np
#create data
df=pd.DataFrame(np.arange(32).reshape((8,4)))
print(df)

#split data
df1 = df[df>19]
df2 = df[df<19]
df1 = df1.dropna(axis=0,how='any')
df2= df2.dropna(axis=0,how='any')

# get result respectively
df1['Col_sum'] = df1.apply(lambda x: x.sum(), axis=1)
df2['Col_sum'] = df2.apply(lambda x: x.sum(), axis=1)

ind1 = df1.index.tolist()
ind2 = df2.index.tolist()
df1 = df1['Col_sum'].values
df2 = df2['Col_sum'].values
print(df1.shape)
print(df2.shape)
df3 = pd.DataFrame(df1)
df4 = pd.DataFrame(df2)
print(df3)
print(df4)
df3.index = ind1
df4.index = ind2
print(df3)
print(df4)
# get ordered result
df  =  pd.concat([df3,df4])

df.sort_index(inplace=True)
# df = df.sort_values()
print(df)

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
df1 = df1['Col_sum']
df2 = df2['Col_sum']
print(df1)
print(df2)


# get ordered result
df  =  pd.concat([df1,df2])
df = df.sort_values()
print(df)

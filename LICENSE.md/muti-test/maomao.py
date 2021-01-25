# For Yuxin Tian Using
import numpy as np
import math
import pandas as pd

df1 = pd.read_excel('abn.xlsx', sheet_name="Sheet1")
df1 = df1.fillna(0)
df1 = df1.values

df2 = pd.read_excel('n.xlsx', sheet_name="Sheet1")
df2 = df2.fillna(0)

df2 = df2.values
sum = 0
for i in range(0,df1.shape[0]):
    for j in range(0,df1.shape[1]):
        sum += np.abs(df1[i,j]-df2[i,j])
print(sum)
# print(df2.shape)

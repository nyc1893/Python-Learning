import numpy as np
import pandas as pd


df1 = pd.read_csv("C:/360Downloads/data/2008-2010/whole/Mits_kW_2008.csv")
df1 = df1.iloc[:,5:]
df1[df1<0] = 0

df1['Col_sum'] = df1.apply(lambda x: x.sum()/1000, axis=1)
print(df1.head())
print(df1.shape)

df1['Col_sum'].to_csv("C:/360Downloads/data/2008-2010/whole/total_mit_08.csv",index = None)




df2 = pd.read_csv("C:/360Downloads/data/2008-2010/whole/GE_kW_2008.csv")
df2 = df2.iloc[:,5:]
df2[df2<0] = 0

df2['Col_sum'] = df2.apply(lambda x: x.sum()/1000, axis=1)
print(df2.head())
print(df2.shape)

df2['Col_sum'].to_csv("C:/360Downloads/data/2008-2010/whole/total_ge_08.csv",index = None)

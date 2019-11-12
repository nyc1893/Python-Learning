
import pandas as pd
import numpy as np


df1 = pd.read_csv("C:/360Downloads/data/correct/Mits_kW_2009.csv")    
df2 = pd.read_csv("C:/360Downloads/data/correct/Mits_kW_2010.csv")    

# print(df1.head())

# print(df1.isnull().any(axis=0))

# print(df1.count(axis = 0 ))
# print(df1[df1.count(axis = 1 )>200].shape)
# print(df1[df1.count(axis = 1 )>220].shape)


# print(df2[df2.count(axis = 1 )>200].shape)
# print(df2[df2.count(axis = 1 )>220].shape)



# print(df1[df1.count(axis = 1 )>220])
# print(df1[df1.count(axis = 1 )>220].shape)
# print(df1[df1.count(axis = 1 )>220].shape[0])
# print(df1[df1.count(axis = 1 )>220].shape[1])
# print(df1.shape[0])
"""
"""
rate1 = ( df1[df1.count(axis = 1 )>220].shape[0])/(df1.shape[0])
rate2 = ( df2[df2.count(axis = 1 )>220].shape[0])/(df2.shape[0])
print(df1.shape)
print(df2.shape)
print("Mit 2009 = ",rate1)
print("Mit 2010 = ",rate2)
print(np.where(np.isnan(df1)))
df1 = df1[df1.count(axis = 1 )>220]
df2 = df2[df2.count(axis = 1 )>220]

rate1 = ( df1[df1.count(axis = 1 )>220].shape[0])/(df1.shape[0])
rate2 = ( df2[df2.count(axis = 1 )>220].shape[0])/(df2.shape[0])

print("Mit 2009 = ",rate1)
print("Mit 2010 = ",rate2)
# print(np.where(np.isnan(df1)))
# print(np.where(np.isnan(df2)))
df1.to_csv('mit_2009.csv',index =None)
df2.to_csv('mit_2010.csv',index =None)

print(df1.shape)
print(df2.shape)
df1 = pd.read_csv("C:/360Downloads/data/correct/GE_kW_2009.csv")    
df2 = pd.read_csv("C:/360Downloads/data/correct/GE_kW_2010.csv")    
# print(df1.count(axis = 1))

rate1 = ( df1[df1.count(axis = 1 )>52].shape[0])/(df1.shape[0])
rate2 = ( df2[df2.count(axis = 1 )>52].shape[0])/(df2.shape[0])
print("GE 2009 = ",rate1)
print("GE 2010 = ",rate2)

print(df1.shape)
print(df2.shape)
df1 = df1[df1.count(axis = 1 )>52]
df2 = df2[df2.count(axis = 1 )>52]

rate1 = ( df1[df1.count(axis = 1 )>52].shape[0])/(df1.shape[0])
rate2 = ( df2[df2.count(axis = 1 )>52].shape[0])/(df2.shape[0])
print("GE 2009 = ",rate1)
print("GE 2010 = ",rate2)
df1.to_csv('ge_2009.csv',index =None)
df2.to_csv('ge_2010.csv',index =None)
print(df1.shape)
print(df2.shape)

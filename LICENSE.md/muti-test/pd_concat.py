import numpy as np
import pandas as pd

df1 = {'l0':[1,2,3]}
df1 = pd.DataFrame(df1)



df2 = {'l0':[7,8,9]}
df2 = pd.DataFrame(df2)

df = pd.concat([df1, df2], axis=0)
#axis = 0 means y-axis join
#axis = 1 means x-axis join
df=df.reset_index(drop = True)
print(df)

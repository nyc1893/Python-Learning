#forced to change event into valiation set

import numpy as np
import math
import pandas as pd



arr = np.load("val_12.npy").tolist()
print(arr)
if (86 and 85 not in arr):
    arr.append(85)
    arr.append(86)
arr = np.array(arr)
np.save("val_12.npy", arr)

list =[]
arr = np.load("tr_12.npy").tolist()
print(arr)
for i in range(len(arr)): 
    if(arr[i]!= 86 or arr[i]!= 85 ):
        list.append(arr[i])
        
arr = np.array(list)
np.save("tr_12.npy", arr)        
    

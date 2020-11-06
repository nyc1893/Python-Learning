

import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
import math
from scipy.stats.distributions import chi2
import os
import sys





    

        

def run(k,Chi_window,para_num):   
    path1 = 'parasearch'+str(para_num)
    path2 = '/Chi_window='+str(Chi_window)
    path3 = '/det'+str(k)+'/'
    fileList=os.listdir(path1+path2+path3)
    sum = 0 
    for f in fileList:
        sum = sum +1
        x = f.split(".", 1)
        # print(x[0])
    # print(path3)
    print(str(sum-2))


def main(ii):   

    for i in range(1,7):
        run(i,ii,3)

            
def test():
    fname = '2016_Jan_27_1_Frequency Event'
    df=pq.read_table(path1+fname+'.parquet').to_pandas()  
    deal(df,fname)
    # check(df)
if __name__ == '__main__':  
    ii = int(sys.argv[1])
    main(ii)

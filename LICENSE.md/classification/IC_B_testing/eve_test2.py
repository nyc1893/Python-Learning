# -*- coding: utf-8 -*-
"""
Created on 10/07/2021

@author: Barron
"""

import sys, os
import pandas as pd
import numpy as np
import timeit
import pyarrow.parquet as pq
import math
from scipy.stats.distributions import chi2
from collections import Counter

from datetime import datetime, timedelta

def fun():
    for i in range(8,22):
        if os.path.exists("log/oneday_4_"+str(i)+"_.csv"): 
            df= pd.read_csv("log/oneday_4_"+str(i)+"_.csv")
            dt = df.groupby(["id","time"]).agg(np.sum)
            dy = dt[dt["num"]!=3600]
            # dy = dy.rese
            dy.to_csv("oneday_4_"+str(i)+"_.csv")
            
def main():
    fun()
if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 16:12:02 2020

@author: Hanif , Amir
"""

import sys, os
import pandas as pd
import numpy as np
import timeit
import pyarrow.parquet as pq
import math
import timeit
import pickle
from datetime import datetime, timedelta


def see():
    df = pd.read_csv("2016-check.csv")
    dt = df["0"].unique()
    dt = pd.DataFrame(dt)

    dt.to_csv("temp.csv",index = None)
    print("save done")

    
def main():
    s1 = timeit.default_timer()

    see()
    
    
    s2 = timeit.default_timer()
    print('Time:(min) ', (s2 - s1)/60 )
if __name__ == "__main__":
    main()
    


    
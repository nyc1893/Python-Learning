#This one will get rid of Prefix like P1_...
#!/usr/bin/env python
# coding=utf-8
import os
import numpy as np
import pandas as pd

path = "../data/"

add_list = [
"上海",
"云南",
"全国",
"内蒙古",
"北京",
"吉林",
"四川",
"天津",
"宁夏",
"安徽",
"山东",
"山西",
"广东",
"广西",
"新疆",
"江苏",
"江西",
"河北",
"河南",
"浙江",
"海南",
"湖北",
"湖南",
"甘肃",
"福建",
"西藏",
"贵州",
"辽宁",
"重庆",
"陕西",
"青海",
"黑龙江" ]    



import econci

    
def fun(year):
    # year = 2002
    df = pd.read_csv(path + "common_"+str(year)+".csv",encoding="gb2312")
    comp = econci.Complexity(df, c='主体名称', p='商品名称', values='金额(美元)')
    comp.calculate_indexes()
    
    eci = comp.eci
    pci = comp.pci

    div = comp.diversity
    ubi = comp.ubiquity
    rca = comp.rca
    pro = comp.proximity
    den = comp.density
    dis = comp.distance

    pci.to_csv(str(year)+"-pci.csv",index = None,encoding="gb2312")
    eci.to_csv(str(year)+"-eci.csv",index = None,encoding="gb2312")
    
    div.to_csv(str(year)+"-diversity.csv",encoding="gb2312")
    ubi.to_csv(str(year)+"-ubiquity.csv",encoding="gb2312")    
    rca.to_csv(str(year)+"-rca.csv",encoding="gb2312")
    pro.to_csv(str(year)+"-proximity.csv",encoding="gb2312")
    den.to_csv(str(year)+"-density.csv",encoding="gb2312")
    dis.to_csv(str(year)+"-distance.csv",encoding="gb2312")




def main():

    for i in range(2002, 2021):
        fun(i)


if __name__ == "__main__":
    main()

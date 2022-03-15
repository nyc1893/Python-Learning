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

    
def cal_rca(year):
    # year = 2002
    df = pd.read_csv(path + "common_"+str(year)+".csv",encoding="gb2312")
    
    df = df[["主体名称", "商品名称", "金额(美元)"]]
    df =df[df["主体名称"]!="全国"]
    
    # 东部
    reg1 = ["北京", "天津",   "河北",  "上海", "江苏",  "浙江", "福建", "山东", "广东",  "海南"]
    # reg1= ["北京"]
    # 中部
    reg2 = ["山西",   "安徽",  "江西",  "河南",  "湖北", "湖南"]
    
    # 西部
    reg3 = ["内蒙古",  "广西",  "重庆",  "四川",  "贵州",  "云南",  "西藏",
    "山西",  "甘肃",  "青海",  "宁夏",   "新疆"]

    # 东北
    reg4 = ["黑龙江",  "吉林",   "辽宁"]
    
    # dt = df.values
    # res = []
    # for i in range(df.shape[0]):
    #     if(dt[i,0] in reg1):
    #         res.append("东部")
    #     elif(dt[i,0] in reg2):
    #         res.append("中部")            
    #     elif(dt[i,0] in reg3):
    #         res.append("西部")    
    #     elif(dt[i,0] in reg4):
    #         res.append("东北")                
    #     else:
    #         res.append("无归类")
    # df["reg"] = res
    # print(df.head())
    
    print(df.shape)
    df = df[df["主体名称"].isin(reg1)]
    print(df.shape)
    df= df.reset_index()



    df = df.reset_index()
    print(df.head())    
    print(df.tail()) 

# def calcu():
    comp = econci.Complexity(df, c='主体名称', p='商品名称', values='金额(美元)')
    # comp = econci.Complexity(df, p='商品名称', values='金额(美元)')
    comp.calculate_indexes()
    
    # eci = comp.eci
    # pci = comp.pci

    # div = comp.diversity
    # ubi = comp.ubiquity
    rca = comp.rca
    # pro = comp.proximity
    # den = comp.density
    # dis = comp.distance

    # pci.to_csv(str(year)+"-pci.csv",index = None,encoding="gb2312")
    # eci.to_csv(str(year)+"-eci.csv",index = None,encoding="gb2312")
    
    # div.to_csv(str(year)+"-diversity.csv",encoding="gb2312")
    # ubi.to_csv(str(year)+"-ubiquity.csv",encoding="gb2312")    
    rca.to_csv(str(year)+"-rca-4region.csv",encoding="gb2312")
    # pro.to_csv(str(year)+"-proximity.csv",encoding="gb2312")
    # den.to_csv(str(year)+"-density.csv",encoding="gb2312")
    # dis.to_csv(str(year)+"-distance.csv",encoding="gb2312")

def cal_proximity(year):
    # year = 2002
    df = pd.read_csv(path + "common_"+str(year)+".csv",encoding="gb2312")
    
    df = df[["主体名称", "商品名称", "金额(美元)"]]
    df =df[df["主体名称"]!="全国"]
    
    # 东部
    reg1 = ["北京", "天津",   "河北",  "上海", "江苏",  "浙江", "福建", "山东", "广东",  "海南"]
    # reg1 = ["北京"]
    # 中部
    # reg2 = ["山西",   "安徽",  "江西",  "河南",  "湖北", "湖南"]
    
    # 西部
    # reg3 = ["内蒙古",  "广西",  "重庆",  "四川",  "贵州",  "云南",  "西藏",
    # "山西",  "甘肃",  "青海",  "宁夏",   "新疆"]

    # 东北
    # reg4 = ["黑龙江",  "吉林",   "辽宁"]
    
    
    print(df.shape)
    df = df[df["主体名称"].isin(reg1)]
    print(df.shape)
    df= df.reset_index()
    print(df.head())
    # dt = df.values
    # res = []
    # for i in range(df.shape[0]):
        # if(dt[i,0] in reg1):
            # res.append("东部")
        # elif(dt[i,0] in reg2):
            # res.append("中部")            
        # elif(dt[i,0] in reg3):
            # res.append("西部")    
        # elif(dt[i,0] in reg4):
            # res.append("东北")                
        # else:
            # res.append("无归类")
    # df["reg"] = res
    # df = df[df["reg"]=="东部"]
    
    # df.pop("主体名称")
    # 
    # print(df.head())
    

    comp = econci.Complexity(df, c='主体名称', p='商品名称', values='金额(美元)')
    comp.calculate_indexes()
    


    pro = comp.proximity



    pro.to_csv(str(year)+"-proximity1.csv",encoding="gb2312")





def main():
    ll = [2002,2010,2019]
    for i in ll:
        # cal_proximity(i)
        cal_proximity(i)
        cal_rca(i)
if __name__ == "__main__":
    main()

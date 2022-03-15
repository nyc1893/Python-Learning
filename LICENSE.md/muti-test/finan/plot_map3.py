
    
from pyvis.network import Network
import networkx as nx
import numpy as np
import pandas as pd

# input a string and return a different label

def guilei(word):
    label_list=["食品","shida"]
    temp = word.split("_")
    value = int(temp[1])
    if(value<=5 and value>0 ):
        return 1

    elif(value<10 and value>5 ):
        return 2

    else:
        return  3

def graph02(v):
    nx_graph = nx.cycle_graph(1)
    # nx_graph.nodes[0]['title'] = 'Number 1'
    # nx_graph.nodes[0]['group'] = 1

    df = pd.read_csv("2002-rca-4region.csv",encoding="gb2312")
    mean_df = df.mean()
    df = pd.DataFrame(mean_df)
    df.columns =["vv"]
    df["ind"] = df.index
    print(df.head())
    df = df.values
    for i in range(df.shape[0]):
        nx_graph.add_node(df[i,1], size=df[i,0]*10, title='couple', group=2)
    # 
    # nx_graph.add_node("C2", size=15, title='couple', group=2)
    # nx_graph.add_node(20, size=15, title='couple', group=2)
    df = pd.read_csv("2002-proximity1.csv",encoding="gb2312")
    rowname = df.pop("商品名称").values
    # rowname= df["商品名称"]
    col= df.columns.values

    print(type(rowname))
    print(type(col))

    print(col)
    print(rowname)
    print(df.head())
    df = df.values

    # for j in range(0,rowname.shape[0]):
    #     for i in range(0,col.shape[0]-1):
    #         # print(df[j][i])
    #         if(df[j][i]>v):
    #             nx_graph.add_edge(rowname[j], col[i], weight=100)

    # print()
    nx_graph.add_edge("C1", 20, weight=5)
    nx_graph.add_edge("C1", "C2", weight=100)
    nx_graph.add_node(25, size=25, label='lonely', title='lonely node', group=3)
    nt = Network('800px', '800px')

    nt.from_nx(nx_graph)
    nt.show('nx02_'+str(int(v*100))+'.html')



def graph10(v):
    nx_graph = nx.cycle_graph(1)
    # nx_graph.nodes[0]['title'] = 'Number 1'
    # nx_graph.nodes[0]['group'] = 1

    df = pd.read_csv("2010-rca-4region.csv",encoding="gb2312")
    mean_df = df.mean()
    df = pd.DataFrame(mean_df)
    df.columns =["vv"]
    df["ind"] = df.index
    print(df.head())
    df = df.values
    for i in range(df.shape[0]):
        nx_graph.add_node(df[i,1], size=df[i,0]*10, title='couple', group=2)
    # 
    # nx_graph.add_node("C2", size=15, title='couple', group=2)
    # nx_graph.add_node(20, size=15, title='couple', group=2)
    df = pd.read_csv("2010-proximity1.csv",encoding="gb2312")
    rowname = df.pop("商品名称").values
    # rowname= df["商品名称"]
    col= df.columns.values

    print(type(rowname))
    print(type(col))

    print(col)
    print(rowname)
    print(df.head())
    df = df.values

    for j in range(0,rowname.shape[0]):
        for i in range(0,col.shape[0]-1):
            # print(df[j][i])
            if(df[j][i]>v):
                nx_graph.add_edge(rowname[j], col[i], weight=100)

    # print()
    # nx_graph.add_edge("C1", 20, weight=5)
    # nx_graph.add_edge("C1", "C2", weight=100)
    # nx_graph.add_node(25, size=25, label='lonely', title='lonely node', group=3)
    nt = Network('800px', '800px')

    nt.from_nx(nx_graph)
    nt.show('nx10_'+str(v*100)+'.html')


def graph19(v):
    nx_graph = nx.cycle_graph(1)
    # nx_graph.nodes[0]['title'] = 'Number 1'
    # nx_graph.nodes[0]['group'] = 1

    df = pd.read_csv("2019-rca-4region.csv",encoding="gb2312")
    mean_df = df.mean()
    df = pd.DataFrame(mean_df)
    df.columns =["vv"]
    df["ind"] = df.index
    print(df.head())
    df = df.values
    for i in range(df.shape[0]):
        nx_graph.add_node(df[i,1], size=df[i,0]*10, title='couple', group=2)
    # 
    # nx_graph.add_node("C2", size=15, title='couple', group=2)
    # nx_graph.add_node(20, size=15, title='couple', group=2)
    df = pd.read_csv("2019-proximity1.csv",encoding="gb2312")
    rowname = df.pop("商品名称").values
    # rowname= df["商品名称"]
    col= df.columns.values

    print(type(rowname))
    print(type(col))

    print(col)
    print(rowname)
    print(df.head())
    df = df.values

    for j in range(0,rowname.shape[0]):
        for i in range(0,col.shape[0]-1):
            # print(df[j][i])
            if(df[j][i]>v):
                nx_graph.add_edge(rowname[j], col[i], weight=100)

    # print()
    # nx_graph.add_edge("C1", 20, weight=5)
    # nx_graph.add_edge("C1", "C2", weight=100)
    # nx_graph.add_node(25, size=25, label='lonely', title='lonely node', group=3)
    nt = Network('800px', '800px')

    nt.from_nx(nx_graph)
    nt.show('nx19_'+str(v*100)+'.html')

def main():
    for v in range(15):
        graph02(0.01*v+0.55)
        graph10(0.01*v+0.55)
        graph19(0.01*v+0.55)

def test():
    guilei("C_28")


graph02(0.55)




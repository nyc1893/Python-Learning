


from pyecharts.charts import Geo
from pyecharts.globals import GeoType
from pyecharts import options as opts
import pandas as pd
import numpy as np


def pt_2002():
    path = "output/"
    df = pd.read_csv(path + "2002-eci.csv",encoding="gb2312")
    df =df[df["主体名称"]!="全国"]
    regions = df["主体名称"].values.tolist()
    values = df["eci"].values.tolist()


    max_v = round(max(values),3)+0.1
    min_v = round(min(values),3)-0.1
    
    
    # max_v = int(max(values))+1
    # min_v = int(min(values))
    print(df.head())


    # regions = ['北京','上海','天津','重庆','广东','深圳','杭州','南京','四川','武汉','西安','郑州','厦门']
    # values = [94, 98, 76, 89, 65, 64, 56, 59, 45, 23, 22, 22, 21]#随便输入的数据
    g = (Geo()
            .add_schema(maptype="china")
            .add("geo", zip(regions, values), type_ = GeoType.EFFECT_SCATTER)
            .set_series_opts(label_opts=opts.LabelOpts(is_show=False))
            .set_global_opts(

                visualmap_opts=opts.VisualMapOpts(is_piecewise=True,max_=max_v,min_=min_v),
                title_opts=opts.TitleOpts(title="2002年 ECI 指标"))
             .render('ECI-2002.html')
        )
        
        
def pt_2010():
    path = "output/"
    df = pd.read_csv(path + "2010-eci.csv",encoding="gb2312")
    df =df[df["主体名称"]!="全国"]
    regions = df["主体名称"].values.tolist()
    values = df["eci"].values.tolist()


    max_v = round(max(values),3)+0.1
    min_v = round(min(values),3)-0.1
    
    
    # max_v = int(max(values))+1
    # min_v = int(min(values))
    print(df.head())


    # regions = ['北京','上海','天津','重庆','广东','深圳','杭州','南京','四川','武汉','西安','郑州','厦门']
    # values = [94, 98, 76, 89, 65, 64, 56, 59, 45, 23, 22, 22, 21]#随便输入的数据
    g = (Geo()
            .add_schema(maptype="china")
            .add("geo", zip(regions, values), type_ = GeoType.EFFECT_SCATTER)
            .set_series_opts(label_opts=opts.LabelOpts(is_show=False))
            .set_global_opts(

                visualmap_opts=opts.VisualMapOpts(is_piecewise=True,max_=max_v,min_=min_v),
                title_opts=opts.TitleOpts(title="2010年 ECI 指标"))
             .render('ECI-2010.html')
        )
        
def pt_2019():
    path = "output/"
    df = pd.read_csv(path + "2019-eci.csv",encoding="gb2312")
    df =df[df["主体名称"]!="全国"]
    regions = df["主体名称"].values.tolist()
    values = df["eci"].values.tolist()


    max_v = round(max(values),3)+0.1
    min_v = round(min(values),3)-0.1
    
    
    # max_v = int(max(values))+1
    # min_v = int(min(values))
    print(regions)


    # regions = ['北京','上海','天津','重庆','广东','深圳','杭州','南京','四川','武汉','西安','郑州','厦门']
    # values = [94, 98, 76, 89, 65, 64, 56, 59, 45, 23, 22, 22, 21]#随便输入的数据
    g = (Geo()
            .add_schema(maptype="china")
            .add("geo", zip(regions, values), type_ = GeoType.EFFECT_SCATTER)
            .set_series_opts(label_opts=opts.LabelOpts(is_show=False))
            .set_global_opts(

                visualmap_opts=opts.VisualMapOpts(is_piecewise=True,max_=max_v,min_=min_v),
                title_opts=opts.TitleOpts(title="2019年 ECI 指标"))
             .render('ECI-2019.html')
        )
        
        
        

        
def pt_02():
    path = "output/"
    df = pd.read_csv(path + "2002-density.csv",encoding="gb2312")
    df =df[df["主体名称"]!="全国"]
    print(df.shape)
    temp = df.iloc[:,1:]
    df["avg"] = temp.sum(axis=1)/temp.shape[1]
    print(temp.shape)
    
    
    df = df[["主体名称","avg"]]
    regions = df["主体名称"].values.tolist()
    values = df["avg"].values.tolist()


    max_v = max(values)
    min_v = min(values)
    print(df.head())
    

    # regions = ['北京','上海','天津','重庆','广东','深圳','杭州','南京','四川','武汉','西安','郑州','厦门']
    # values = [94, 98, 76, 89, 65, 64, 56, 59, 45, 23, 22, 22, 21]#随便输入的数据
    g = (Geo()
            .add_schema(maptype="china")
            .add("geo", zip(regions, values), type_ = GeoType.EFFECT_SCATTER)
            .set_series_opts(label_opts=opts.LabelOpts(is_show=False))
            .set_global_opts(

                visualmap_opts=opts.VisualMapOpts(is_piecewise=True,max_=max_v,min_=min_v),
                title_opts=opts.TitleOpts(title="2002年 average density 指标"))
             .render('density-2002.html')
        )
        
def pt_10():
    path = "output/"
    df = pd.read_csv(path + "2010-density.csv",encoding="gb2312")
    df =df[df["主体名称"]!="全国"]
    print(df.shape)
    temp = df.iloc[:,1:]
    df["avg"] = temp.sum(axis=1)/temp.shape[1]
    print(temp.shape)
    
    
    df = df[["主体名称","avg"]]
    regions = df["主体名称"].values.tolist()
    values = df["avg"].values.tolist()


    max_v = max(values)
    min_v = min(values)
    print(df.head())
    

    # regions = ['北京','上海','天津','重庆','广东','深圳','杭州','南京','四川','武汉','西安','郑州','厦门']
    # values = [94, 98, 76, 89, 65, 64, 56, 59, 45, 23, 22, 22, 21]#随便输入的数据
    g = (Geo()
            .add_schema(maptype="china")
            .add("geo", zip(regions, values), type_ = GeoType.EFFECT_SCATTER)
            .set_series_opts(label_opts=opts.LabelOpts(is_show=False))
            .set_global_opts(

                visualmap_opts=opts.VisualMapOpts(is_piecewise=True,max_=max_v,min_=min_v),
                title_opts=opts.TitleOpts(title="2010年 average density 指标"))
             .render('density-2010.html')
        )
        
        
def pt_19():
    path = "output/"
    df = pd.read_csv(path + "2019-density.csv",encoding="gb2312")
    df =df[df["主体名称"]!="全国"]
    print(df.shape)
    temp = df.iloc[:,1:]
    df["avg"] = temp.sum(axis=1)/temp.shape[1]
    print(temp.shape)
    
    
    df = df[["主体名称","avg"]]
    regions = df["主体名称"].values.tolist()
    values = df["avg"].values.tolist()


    max_v = max(values)
    min_v = min(values)
 
    

    # regions = ['北京','上海','天津','重庆','广东','深圳','杭州','南京','四川','武汉','西安','郑州','厦门']
    # values = [94, 98, 76, 89, 65, 64, 56, 59, 45, 23, 22, 22, 21]#随便输入的数据
    g = (Geo()
            .add_schema(maptype="china")
            .add("geo", zip(regions, values), type_ = GeoType.EFFECT_SCATTER)
            .set_series_opts(label_opts=opts.LabelOpts(is_show=False))
            .set_global_opts(

                visualmap_opts=opts.VisualMapOpts(is_piecewise=True,max_=max_v,min_=min_v),
                title_opts=opts.TitleOpts(title="2019年 average density 指标"))
             .render('density-2019.html')
        )
        
        
        
pt_2002()
pt_2010()
pt_2019()

# pt_02()
# pt_10()
# pt_19()
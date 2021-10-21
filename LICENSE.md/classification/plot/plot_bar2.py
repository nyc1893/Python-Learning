import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
# 这两行代码解决 plt 中文显示的问题

plt.figure( figsize=(8,4))
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.yticks(fontproperties = 'Times New Roman', size = 18)
plt.xticks(fontproperties = 'Times New Roman', size = 18)
# 输入统计数据

font = {'family':'Times New roman'  #'serif', 
#         ,'style':'italic'
        ,'weight':'normal'  # 'bold' 
#         ,'color':'red'
        ,'size':18
       }

font2 = {'family':'Times New roman'  #'serif', 
#         ,'style':'italic'
        ,'weight':'normal'  # 'normal' 
#         ,'color':'red'
        ,'size':18
       }    

def acc():
    waters = ('Testing Data', 'Estimated Label')
    c1 = [94, 100]
    c2 = [87, 88]
    c3 = [87,82]
    c4 = [84,88]
    c5 = [84,79]
    c6 = [83, 89]
    c7 = [80, 74]
    c8 = [80,89]
    c9 = [78,72]


    n = 100
    y = np.arange(0, n, step=.5)

     
    # y 的值归一化到[0, 1]
    # 因为 y 大到一定程度超过临界数值后颜色就会饱和不变(不使用循环colormap)。
    norm = plt.Normalize(y.min(), y.max())
    # matplotlib.colors.Normalize 对象，可以作为参数传入到绘图方法里
    # 也可给其传入数值直接计算归一化的结果
    norm_y = norm(y)
    map1 = cm.get_cmap(name='winter')
    map2 = cm.get_cmap(name='PiYG')
    map3 = cm.get_cmap(name='binary')
    # plasma
    # viridis
    color = map1(norm_y)


    bar_width = 0.1  # 条形宽度
    ind1 = np.arange(len(waters))  # 男生条形图的横坐标
    ind2 = ind1 + 1*bar_width  # 女生条形图的横坐标
    ind3 = ind1 + 2*bar_width  
    ind4 = ind1 + 3*bar_width  
    ind5 = ind1 + 4*bar_width  
    ind6 = ind1 + 5*bar_width  
    ind7 = ind1 + 6*bar_width  
    ind8 = ind1 + 7*bar_width  
    ind9 = ind1 + 8*bar_width  

    bar_width2 = 0.085
    # 使用两次 bar 函数画出两组条形图
    plt.bar(ind1, height=c1, width=bar_width2,  label='Theoretical limit',color = map1(0.2))
    plt.bar(ind2, height=c2, width=bar_width2,  label='60% Unlabeled Weakly',color = map2(0))
    plt.bar(ind3, height=c3, width=bar_width2,  label='60% Unlabeled Semi',color = map2(0.99))
    plt.bar(ind4, height=c4, width=bar_width2,  label='70% Unlabeled Weakly',color = map2(0.07))
    plt.bar(ind5, height=c5, width=bar_width2,  label='70% Unlabeled Semi',color = map2(0.93))
    plt.bar(ind6, height=c6, width=bar_width2,  label='80% Unlabeled Weakly',color = map2(0.15))
    plt.bar(ind7, height=c7, width=bar_width2,  label='80% Unlabeled Semi',color = map2(0.85))
    plt.bar(ind8, height=c8, width=bar_width2,  label='90% Unlabeled Weakly',color = map2(0.25))
    plt.bar(ind9, height=c9, width=bar_width2,  label='90% Unlabeled Semi',color = map2(0.75))





    plt.legend(loc=[1, 0],prop = font)

    plt.xticks(ind1 + 3.5*bar_width, waters)  # 让横坐标轴刻度显示 waters 里的饮用水， index_male + bar_width/2 为横坐标轴刻度的位置
    plt.ylabel('Accruacy(%)',font2)  # 纵坐标轴标题
    plt.ylim(45,100)
    # plt.ylim(55,90)
    # plt.title('购买饮用水情况的调查结果')  # 图形标题
    plt.tight_layout()
    plt.grid(ls='--')
    plt.show()


def pre():
    waters = ('Testing Data', 'Estimated Label')
    c1 = [95, 100]
    c2 = [91, 50]
    c3 = [90,82]
    c4 = [88,50]
    c5 = [90,79]
    c6 = [87, 50]
    c7 = [82, 74]
    c8 = [82,50]
    c9 = [86,72]


    n = 100
    y = np.arange(0, n, step=.5)

     
    # y 的值归一化到[0, 1]
    # 因为 y 大到一定程度超过临界数值后颜色就会饱和不变(不使用循环colormap)。
    norm = plt.Normalize(y.min(), y.max())
    # matplotlib.colors.Normalize 对象，可以作为参数传入到绘图方法里
    # 也可给其传入数值直接计算归一化的结果
    norm_y = norm(y)
    map1 = cm.get_cmap(name='winter')
    map2 = cm.get_cmap(name='PiYG')
    map3 = cm.get_cmap(name='binary')

    color = map1(norm_y)


    bar_width = 0.1  # 条形宽度
    ind1 = np.arange(len(waters))  # 男生条形图的横坐标
    ind2 = ind1 + 1*bar_width  # 女生条形图的横坐标
    ind3 = ind1 + 2*bar_width  
    ind4 = ind1 + 3*bar_width  
    ind5 = ind1 + 4*bar_width  
    ind6 = ind1 + 5*bar_width  
    ind7 = ind1 + 6*bar_width  
    ind8 = ind1 + 7*bar_width  
    ind9 = ind1 + 8*bar_width  

    bar_width2 = 0.085
    # 使用两次 bar 函数画出两组条形图
    plt.bar(ind1, height=c1, width=bar_width2,  label='Theoretical limit',color = map1(0.2))
    plt.bar(ind2, height=c2, width=bar_width2,  label='60% Unlabeled Weakly',color = map2(0))
    plt.bar(ind3, height=c3, width=bar_width2,  label='60% Unlabeled Semi',color = map2(0.99))
    plt.bar(ind4, height=c4, width=bar_width2,  label='70% Unlabeled Weakly',color = map2(0.07))
    plt.bar(ind5, height=c5, width=bar_width2,  label='70% Unlabeled Semi',color = map2(0.93))
    plt.bar(ind6, height=c6, width=bar_width2,  label='80% Unlabeled Weakly',color = map2(0.15))
    plt.bar(ind7, height=c7, width=bar_width2,  label='80% Unlabeled Semi',color = map2(0.85))
    plt.bar(ind8, height=c8, width=bar_width2,  label='90% Unlabeled Weakly',color = map2(0.25))
    plt.bar(ind9, height=c9, width=bar_width2,  label='90% Unlabeled Semi',color = map2(0.75))





    plt.legend(loc=[1, 0],prop = font)

    plt.xticks(ind1 + 3.5*bar_width, waters)  # 让横坐标轴刻度显示 waters 里的饮用水， index_male + bar_width/2 为横坐标轴刻度的位置
    plt.ylabel('Precision(%)',font2)  # 纵坐标轴标题
    plt.ylim(45,100)
    # plt.ylim(55,90)
    # plt.title('购买饮用水情况的调查结果')  # 图形标题
    plt.tight_layout()
    plt.grid(ls='--')
    plt.show()

def recall():
    waters = ('Testing Data', 'Estimated Label')
    c1 = [88, 100]
    c2 = [80, 50]
    c3 = [77,70]
    c4 = [75,50]
    c5 = [74,67]
    c6 = [72, 50]
    c7 = [68, 61]
    c8 = [68,50]
    c9 = [63,56]


    n = 100
    y = np.arange(0, n, step=.5)

     
    # y 的值归一化到[0, 1]
    # 因为 y 大到一定程度超过临界数值后颜色就会饱和不变(不使用循环colormap)。
    norm = plt.Normalize(y.min(), y.max())
    # matplotlib.colors.Normalize 对象，可以作为参数传入到绘图方法里
    # 也可给其传入数值直接计算归一化的结果
    norm_y = norm(y)
    map1 = cm.get_cmap(name='winter')
    map2 = cm.get_cmap(name='PiYG')
    map3 = cm.get_cmap(name='binary')

    color = map1(norm_y)


    bar_width = 0.1  # 条形宽度
    ind1 = np.arange(len(waters))  # 男生条形图的横坐标
    ind2 = ind1 + 1*bar_width  # 女生条形图的横坐标
    ind3 = ind1 + 2*bar_width  
    ind4 = ind1 + 3*bar_width  
    ind5 = ind1 + 4*bar_width  
    ind6 = ind1 + 5*bar_width  
    ind7 = ind1 + 6*bar_width  
    ind8 = ind1 + 7*bar_width  
    ind9 = ind1 + 8*bar_width  

    bar_width2 = 0.085
    # 使用两次 bar 函数画出两组条形图
    plt.bar(ind1, height=c1, width=bar_width2,  label='Theoretical limit',color = map1(0.2))
    plt.bar(ind2, height=c2, width=bar_width2,  label='60% Unlabeled Weakly',color = map2(0))
    plt.bar(ind3, height=c3, width=bar_width2,  label='60% Unlabeled Semi',color = map2(0.99))
    plt.bar(ind4, height=c4, width=bar_width2,  label='70% Unlabeled Weakly',color = map2(0.07))
    plt.bar(ind5, height=c5, width=bar_width2,  label='70% Unlabeled Semi',color = map2(0.93))
    plt.bar(ind6, height=c6, width=bar_width2,  label='80% Unlabeled Weakly',color = map2(0.15))
    plt.bar(ind7, height=c7, width=bar_width2,  label='80% Unlabeled Semi',color = map2(0.85))
    plt.bar(ind8, height=c8, width=bar_width2,  label='90% Unlabeled Weakly',color = map2(0.25))
    plt.bar(ind9, height=c9, width=bar_width2,  label='90% Unlabeled Semi',color = map2(0.75))





    plt.legend(loc=[1, 0],prop = font)

    plt.xticks(ind1 + 3.5*bar_width, waters)  # 让横坐标轴刻度显示 waters 里的饮用水， index_male + bar_width/2 为横坐标轴刻度的位置
    plt.ylabel('Recall(%)',font2)  # 纵坐标轴标题
    plt.ylim(45,100)
    # plt.ylim(55,90)
    # plt.title('购买饮用水情况的调查结果')  # 图形标题
    plt.tight_layout()
    plt.grid(ls='--')
    plt.show()

def F1():
    waters = ('Testing Data', 'Estimated Label')
    c1 = [91, 100]
    c2 = [83, 50]
    c3 = [81,75]
    c4 = [78,50]
    c5 = [78,71]
    c6 = [75, 50]
    c7 = [70, 63]
    c8 = [70,50]
    c9 = [63,56]


    n = 100
    y = np.arange(0, n, step=.5)

     
    # y 的值归一化到[0, 1]
    # 因为 y 大到一定程度超过临界数值后颜色就会饱和不变(不使用循环colormap)。
    norm = plt.Normalize(y.min(), y.max())
    # matplotlib.colors.Normalize 对象，可以作为参数传入到绘图方法里
    # 也可给其传入数值直接计算归一化的结果
    norm_y = norm(y)
    map1 = cm.get_cmap(name='winter')
    map2 = cm.get_cmap(name='PiYG')
    map3 = cm.get_cmap(name='binary')

    color = map1(norm_y)


    bar_width = 0.1  # 条形宽度
    ind1 = np.arange(len(waters))  # 男生条形图的横坐标
    ind2 = ind1 + 1*bar_width  # 女生条形图的横坐标
    ind3 = ind1 + 2*bar_width  
    ind4 = ind1 + 3*bar_width  
    ind5 = ind1 + 4*bar_width  
    ind6 = ind1 + 5*bar_width  
    ind7 = ind1 + 6*bar_width  
    ind8 = ind1 + 7*bar_width  
    ind9 = ind1 + 8*bar_width  

    bar_width2 = 0.085
    # 使用两次 bar 函数画出两组条形图
    plt.bar(ind1, height=c1, width=bar_width2,  label='Theoretical limit',color = map1(0.2))
    plt.bar(ind2, height=c2, width=bar_width2,  label='60% Unlabeled Weakly',color = map2(0))
    plt.bar(ind3, height=c3, width=bar_width2,  label='60% Unlabeled Semi',color = map2(0.99))
    plt.bar(ind4, height=c4, width=bar_width2,  label='70% Unlabeled Weakly',color = map2(0.07))
    plt.bar(ind5, height=c5, width=bar_width2,  label='70% Unlabeled Semi',color = map2(0.93))
    plt.bar(ind6, height=c6, width=bar_width2,  label='80% Unlabeled Weakly',color = map2(0.15))
    plt.bar(ind7, height=c7, width=bar_width2,  label='80% Unlabeled Semi',color = map2(0.85))
    plt.bar(ind8, height=c8, width=bar_width2,  label='90% Unlabeled Weakly',color = map2(0.25))
    plt.bar(ind9, height=c9, width=bar_width2,  label='90% Unlabeled Semi',color = map2(0.75))





    plt.legend(loc=[1, 0],prop = font)

    plt.xticks(ind1 + 3.5*bar_width, waters)  # 让横坐标轴刻度显示 waters 里的饮用水， index_male + bar_width/2 为横坐标轴刻度的位置
    plt.ylabel('F1 score (%)',font2)  # 纵坐标轴标题
    plt.ylim(45,100)
    # plt.ylim(55,90)
    # plt.title('购买饮用水情况的调查结果')  # 图形标题
    plt.tight_layout()
    plt.grid(ls='--')
    plt.show()
    
acc()
# pre()
# recall()
# F1()
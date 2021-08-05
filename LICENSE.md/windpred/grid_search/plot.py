
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
df = pd.read_csv("df.csv")
df.index = pd.Series(list(range(92,99)))

# df = pd.read_csv("df2.csv")
# df.index = pd.Series(['55', '65', '75', '85', '95'])

plt.figure(dpi=120)


plt.rc('font', family='Times New Roman')
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
# 绘制热力图
h = sns.heatmap(

            data=df,#矩阵数据集，数据的index和columns分别为heatmap的y轴方向和x轴方向标签


            cmap='hot_r', # 指定填充色'PuBuGn'，，jet
            # center=1,
            linewidths=.1, # 设置每个单元格边框的宽度
            linecolor="black",
            annot=True,  # 显示数值
            cbar=False,

                 vmin=1.90,#图例（右侧颜色条color bar）中最小显示值 
                 vmax=2.81,#图例（右侧颜色条color bar）中最大显示值
                 # cmap= "YlGnBu"
           )
cb = h.figure.colorbar(h.collections[0]) #显示colorbar
# cb.collections[0].colorbar.set_label("Hello")
cb.ax.tick_params(labelsize=14)  # 设置colorbar刻度字体大小。

# 添加标题, fontweight='bold'
# plt.title('NMAE of Global View', fontsize=20) 
plt.title('NMAE of Zoom In View', fontsize=20) 
plt.xticks(fontsize=11)
plt.yticks(fontsize=11)
plt.yticks(rotation=0)
plt.xlabel('Window Size', fontsize=16)  # 经度
plt.ylabel('Quntile(%)', fontsize=16)   # 纬度
plt.rcParams['savefig.dpi'] = 600
plt.tight_layout()

plt.show()
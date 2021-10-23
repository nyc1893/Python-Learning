# from sklearn.datasets import make_moons
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import pickle as pickle  
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support
import warnings
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

warnings.filterwarnings('ignore')

# Random Forest Classifier  
def random_forest_classifier(train_x, train_y):  
    from sklearn.ensemble import RandomForestClassifier  
    model = RandomForestClassifier(n_estimators=100)  
    model.fit(train_x, train_y)  
    return model  
from sklearn.utils import shuffle

def filter_data():
    path = "data/"
    df2 = pd.read_csv(path+"X_train.csv")
    df4 = df2[df2["label"] == 0]
    df5 = df2[df2["label"] == 2]
    
    df1 = df2[df2["label"] == 1]
    print(df1.shape)
    # df3 = df1[(df1["max_v_dup"] > 0.01) | (df1["max_v_ddn"] >0.005)
    # | (df1["max_v_adn"] > 2)| (df1["max_v_aup"] > 0.07)]
    
    # df4 = df4[(df4["max_v_dup"] < 0.01) | (df4["max_v_ddn"] <0.005)
    # | (df4["max_v_adn"] <2)| (df4["max_v_aup"] < 0.07)]
    df3 = df1
    df  = pd.concat([df4, df5])
    df  = pd.concat([df, df3])   
    df = shuffle(df)
    X_train = df
    df2 = pd.read_csv(path+"X_val.csv")
    df4 = df2[df2["label"] == 0]
    df5 = df2[df2["label"] == 2]
    
    df1 = df2[df2["label"] == 1]
    print(df1.shape)
    # df3 = df1[(df1["max_v_dup"] > 0.01) | (df1["max_v_ddn"] >0.005)
    # | (df1["max_v_adn"] > 2)| (df1["max_v_aup"] > 0.07)]

    # df4 = df4[(df4["max_v_dup"] < 0.01) | (df4["max_v_ddn"] <0.005)
    # | (df4["max_v_adn"] <2)| (df4["max_v_aup"] < 0.07)]
    
    df3 = df1
    df  = pd.concat([df4, df5])
    df  = pd.concat([df, df3])   
    df = shuffle(df)
    X_test = df
    
    return X_train,X_test
    
# def detrans(y):
    # yy = np.copy(y)
    # yy[yy==1] = 0
    # yy[yy==2] = 1
    # yy[yy==3] = 2
    # return yy

# python wk2.py 2>&1 | tee sno.log
def get_sno_label():
    tr = pd.read_csv('data/wk_1_tr2.csv')
    test = pd.read_csv('data/wk_1_test2.csv')
    tr.pop("real")
    test.pop("real")
    
    # tr2 = pd.read_csv('data/wk_1_tr2.csv')
    # test2 = pd.read_csv('data/wk_1_test2.csv')
    # tr2.pop("real")
    # test2.pop("real")

    # tr = pd.concat([tr,tr2],axis=1)
    # test = pd.concat([test,test2],axis=1)   

    tr2 = pd.read_csv('data/wk_1_tr.csv')
    test2 = pd.read_csv('data/wk_1_test.csv')
    tr2.pop("real")
    test2.pop("real")
    
    tr = pd.concat([tr,tr2],axis=1)
    test = pd.concat([test,test2],axis=1)  

    # tr2 = pd.read_csv('data/wk_1_tr4.csv')
    # test2 = pd.read_csv('data/wk_1_test4.csv')
    # tr2.pop("real")
    # test2.pop("real")
    
    # tr = pd.concat([tr,tr2],axis=1)
    # test = pd.concat([test,test2],axis=1)  


    y_tr = pd.read_csv('data/wk_1_tr5.csv').values
    y_test = pd.read_csv('data/wk_1_test5.csv').values
    

    tr = tr.values
    test = test.values

    print(tr.shape)
    print(test.shape)



# def cc():
    # arr = []
    # for i in range(0,test.shape[0]):
        # temp = test[i]
        # arr.append(np.argmax(np.bincount(temp)))

    # arr = np.array(arr)
    # print('arrsize:',arr.shape)
    from snorkel.labeling import LabelModel

    label_model = LabelModel(cardinality= 3, verbose=True)
    label_model.fit(L_train=tr, n_epochs=1500, log_freq=100, seed=123)


    pred = label_model.predict(test)

    label_model_acc = label_model.score(L=test, Y=y_test, tie_break_policy="random")[
        "accuracy"]
    print(f"{'Label Model Accuracy:':<25} {label_model_acc * 100:.1f}%")
    tt = pd.read_csv('data/wk_1_test.csv')
    
    tt['SNO'] = pred
    # tt['Maj'] = arr
    # tt['rf_pred'] = rf_pred
    # tt['real'] = tt["label"]
    # semi = pd.read_csv('semi_res.csv')
    # tt['SelfTrain'] = semi['SelfTrain']
    # tt['LabelSpread'] = semi['LabelSpread']
    tt.to_csv('wk_2_test.csv',index = None)

def get_semi_label():
    tt = pd.read_csv('data/wk_1_test5.csv')
    print(tt.shape)
    semi = pd.read_csv('semi_res.csv')
    print(semi.shape)
    # tt = tt.iloc[-semi.shape[0]:,:]
    print(tt.shape)
    tt['SelfTrain'] = semi['SelfTrain']
    tt.to_csv('wk_2_test.csv',index = None)
# get_semi_label()


def esti():
    df1 =pd.read_csv("data/wk_1_test5.csv").values
    df2 = pd.read_csv("wk_2_test.csv")
    df2 = df2["real"].values
    print("SNO")
    # matrix=confusion_matrix(df1, df2)
    # print(matrix)
    # class_report=classification_report(df1, df2)
    # print(class_report)
    print('Macro Precision: {:.2f}'.format(precision_score(df1, df2, average='macro')))
    print('Macro Recall: {:.2f}'.format(recall_score(df1, df2, average='macro')))
    print('Macro F1-score: {:.2f}\n'.format(f1_score(df1, df2, average='macro')))
    
    df2 = pd.read_csv('semi_res.csv').values
    print("Semi")
    # matrix=confusion_matrix(df1, df2)
    # print(matrix)
    # class_report=classification_report(df1, df2)
    # print(class_report)
    
    print('Macro Precision: {:.2f}'.format(precision_score(df1, df2, average='macro')))
    print('Macro Recall: {:.2f}'.format(recall_score(df1, df2, average='macro')))
    print('Macro F1-score: {:.2f}\n'.format(f1_score(df1, df2, average='macro')))
    
# get_sno_label()
# get_semi_label()
# esti()
# print('wk_2_test update done')




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
    
# def detrans(y):
    # yy = np.copy(y)
    # yy[yy==1] = 0
    # yy[yy==2] = 1
    # yy[yy==3] = 2
    # return yy

# python wk2.py 2>&1 | tee sno.log
def get_sno_label():
    # tr = pd.read_csv('data/wk_1_tr.csv')
    # test = pd.read_csv('data/wk_1_test.csv')

    
    
    # tr2 = pd.read_csv('data/wk_1_tr2.csv')
    # test2 = pd.read_csv('data/wk_1_test2.csv')
    # tr2.pop("real")
    # test2.pop("real")
    # tr = pd.concat([tr,tr2],axis=1)
    # test = pd.concat([test,test2],axis=1)    
    tr = pd.read_csv('data/wk_1_tr2.csv')
    test = pd.read_csv('data/wk_1_test2.csv')
    tr.pop("real")
    test.pop("real")
    

    
    tr2 = pd.read_csv('data/wk_1_tr4.csv')
    test2 = pd.read_csv('data/wk_1_test4.csv')
    tr2.pop("real")
    test2.pop("real")
    
    tr = pd.concat([tr,tr2],axis=1)
    test = pd.concat([test,test2],axis=1)    
    
    tr2 = pd.read_csv('data/wk_1_tr3.csv')
    test2 = pd.read_csv('data/wk_1_test3.csv')
    tr2.pop("real")
    test2.pop("real")
    
    tr = pd.concat([tr,tr2],axis=1)
    test = pd.concat([test,test2],axis=1)   
    
    
    df2 = open("data/temp_data.pickle","rb")
    _,y1,_,y2  = pickle.load(df2)

    # print(type(y2))
    
    y_tr = y1.values
    y_test = y2.values

    tr = tr.values
    test = test.values







    from snorkel.labeling import LabelModel

    label_model = LabelModel(cardinality= 3, verbose=True)
    label_model.fit(L_train=tr, n_epochs=1500, log_freq=100, seed=123,lr=0.05,l2=0.4)


    pred = label_model.predict(test)

    label_model_acc = label_model.score(L=test, Y=y_test, tie_break_policy="random")[
        "accuracy"]
    # print(f"{'Label Model Accuracy:':<25} {label_model_acc * 100:.1f}%")
    
    # test.pop('LR')
    
    # print(test.head())
    model = random_forest_classifier(tr, y_tr) 
    rf_pred = model.predict(test)  
    


    tt = pd.read_csv('semi_res.csv')

    arr = []
    for i in range(0,test.shape[0]):
        temp = tt.iloc[i]
        ty= np.argmax(np.bincount(temp))
        if ty !=-1:
            arr.append(ty)
        else:
            arr.append(0)
    arr = np.array(arr)
    print('arrsize:',arr.shape)
    tt['Maj'] = arr
    tt['SNO'] = rf_pred
    tt.to_csv('wk_2_test.csv',index = None)
    # print(tt.tail())
    # print('wk_2_test update done')


    
def esti2():
    # df1 =pd.read_csv("data/wk_1_test5.csv").values
    df3 = pd.read_csv("wk_2_test.csv")
    ll = ["SNO","semi-DT","semi-GBDT","semi-SVC","semi-LR","semi-KNN","Maj"]
    res = []
    print()
    df2 = open("data/temp_data.pickle","rb")
    X_train,y_train,X_val,y_val = pickle.load(df2)
    
    for ii in range(len(ll)):
        # print(ll[ii])
        df1 = df3[ll[ii]].values
        df2 = y_val.values
        # df1 = df2["SNO"].values
        # res.append(accuracy_score(df1, df2))
        temp = round(accuracy_score(df1, df2), 3)  
        res.append(temp)
        
        
        # print('\nAccuracy: {:.2f}\n'.format(accuracy_score(df1, df2)))
        # print('Macro Precision: {:.2f}'.format(precision_score(df1, df2, average='macro')))
        # print('Macro Recall: {:.2f}'.format(recall_score(df1, df2, average='macro')))
        # print('Macro F1-score: {:.2f}\n'.format(f1_score(df1, df2, average='macro')))
    res= np.array(res)
    # print(res)
    return res
def run():
    get_sno_label()
    return esti2()
def main():   
    run()
if __name__ == '__main__':  

    main()
    # update_complete_random(0.5)





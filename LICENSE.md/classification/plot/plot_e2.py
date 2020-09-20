#  Add code to deal with mutilabel
# from sklearn.datasets import make_moons
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import pickle as pickle  
import pandas as pd
import numpy as np

def removePlanned(X,y):
    """
    THIS FUNCTION REMOVES THE PLANNED EVENTS FROM THE EVENT DATASET
    """
    
    X_new=[]
    y_new=[]
    for i in range(len(y)):
        #print(i)
    
        if y[i]==0:
            y_new.append(0)
            X_new.append(X[i,:,:,:])
    
        elif y[i]==1:
            y_new.append(1)
            X_new.append(X[i,:,:,:])
    
            
        elif y[i]==2:
            y_new.append(2)
            X_new.append(X[i,:,:,:])
        
        elif y[i]==3:
            y_new.append(3)
            X_new.append(X[i,:,:,:])
        

    return  np.array(X_new), np.array(y_new)


def separatePMUs(X,y):
    
    """
    This function separates features and their corresponding labels for each PMU
    to make more events 
    """
    
    num_case=X.shape[0]
    num_pmu=X.shape[1]
    num_sample=X.shape[2]
    X=X.reshape(num_case*num_pmu,num_sample)
    y_new=[]
    for i in range(len(y)):
        if y[i]==0:
            for j in range(num_pmu):
                y_new.append(0)
                
        if y[i]==1:
            for j in range(num_pmu):
                y_new.append(1)
                
        if y[i]==2:
            for j in range(num_pmu):
                y_new.append(2)
                
        if y[i]==3:
            for j in range(num_pmu):
                y_new.append(3)
                
        if y[i]==6:
            for j in range(num_pmu):
                y_new.append(6)        
    
    return X,np.array(y_new)

    
def deal_label(y_test):
    y_test = pd.DataFrame(y_test)
    print(y_test.head())
    y_test.columns = ['event',	'label']
    df = y_test.pop('event')
    # print(df.head())
    df = df.str.split('_')
    df2 = pd.DataFrame(df)
    y_test['year'] = df2['event'].str[0]
    y_test['month'] = df2['event'].str[1]
    y_test['day'] = df2['event'].str[2]
    y_test['no'] = df2['event'].str[3]   

    y_test['new'] = y_test['year'].astype(str).str.cat(y_test['month'].astype(str),sep = '_')
    y_test['new'] = y_test['new'].str.cat(y_test['day'].astype(str),sep = '_')
    y_test['new'] = y_test['new'].str.cat(y_test['no'].astype(str),sep = '_')    

    y_test = y_test[['new','label']]
    
    

    df2 = pd.read_csv('df_.csv')
    list = df2['new'].tolist()
    # list = list[0:5]
    # print(len(list))
    y_test['label'] = y_test['label'].astype("int")
    
    y_test.label[y_test['new'].isin(list)] = 6
    print('y_test.shape',y_test.shape)
    y_test.pop('new')
    return y_test.values
    # print(y_test.loc[40:55])
    # y_test.to_csv('dfkan.csv',index =None)
    # 
    # print(y_test.head())    
    
def read_data():  
    # path2 = '../svm/'
    # df1 = pd.read_csv(path2 +'X_train.csv')
    # df2 = pd.read_csv(path2 +'X_test.csv')
    # df1 = pd.concat([df1,df2])
    name = "S1_vp_m"
    path = '../pickleset/'
    p2 = open(path+ "X2_"+name+"_6.pickle","rb")
    # p2 = open(path+ "X2_S2_rocof_6.pickle","rb")
    pk2 = pickle.load(p2)
    X_test = pk2
    # p2 = open(path+ "y2_S2_rocof_6.pickle","rb")
    p2 = open(path+ "y2_"+name+"_6.pickle","rb")
    
    pk2 = pickle.load(p2)
    y_test = pk2
    y_test = deal_label(y_test)
    
    fps=60
    start_crop=int(fps*60*4)
    stop_crop=int(fps*60*7)

    X_test= X_test[:,:,start_crop:stop_crop,:]    
    X_test,y_test = separatePMUs(X_test,y_test)
    # print(y_test)
    return X_test,y_test

    
    
def main():
    
    read_data()


if __name__ == '__main__':  
    main()


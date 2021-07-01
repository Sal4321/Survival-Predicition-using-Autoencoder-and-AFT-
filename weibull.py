# -*- coding: utf-8 -*-
"""
Created on Sun Mar 21 22:44:41 2021

@author: Salehin
"""
import tensorflow as tf
from lifelines import WeibullAFTFitter
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from lifelines.datasets import load_rossi
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.losses import mse
import numpy as np
from Denoised_Autoencoder import Autoencoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Reshape, Flatten, Input, GaussianNoise
from tensorflow.keras.optimizers import SGD
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import random
import missingno as msno
import seaborn as sns
#Data loading and preprocessing
def preprocessing_gene(data):
    #data2=pd.read_excel('C://Personal//Unf Course Folder//Spring 2021//Applied Predictive Modelling//RNA_seq_2.xlsx')
    data=data.T
    #After r
    sample_names=list(data.index)
    sample_names=sample_names[1:]
    feature=data.iloc[0,:]
    data=data.drop(index='attrib_name')
    data.columns=feature
    #check if we have null values
    isnull=pd.isnull(data)
    countmissing=isnull.sum(axis=0)
    total_missing=countmissing.sum()
    #There are 191 missing values from 
    data=data.drop(['RAC1'],axis=1)
    return data

    #the following lines will convert the given data to a numpyt array and then
    # convert to a dataframe with similiar structure as our gene expression dat
    
    
def seriesoutput(df,column_no):
    data=df.iloc[column_no,:]
    data.drop(data.index[0],axis=0,inplace=True)
    return list(data)



def preprocessing_clinical(data):
    clinc=[]
    column_names=clinical.iloc[:,0]
    for i in range(column_names.size):
        data=seriesoutput(clinical,i)
        clinc.append(data)
    clinc=np.array(clinc)  
    clinc=clinc.T  
    clinc=pd.DataFrame(clinc)
    clinc.columns=clinical.iloc[:,0]    
    clinc=clinc.sort_values(by='attrib_name')
    return clinc
#clinc=clinc.reset_index()
#function for finding out rows that are not in gene expression data
def finder(data1,data2):
    rowlist=[]
    for i in range(len(data1)):
        if data1[i] not in data2.values:
            rowlist.append(i)
    return rowlist       
     
    
     


#some exploratory analysis        
# cat=['attrib_name','histological_type','gender','radiation_therapy','race','ethnicity','overallsurvival']   
# clinc.drop(cat,axis=1,inplace=True)

# dec=clinc.describe()
# clinc=clinc.astype(float)
# desc=clinc.describe()

if __name__ == "__main__":
    #Read in gene expression data
    data=pd.read_excel('C://Personal//Unf Course Folder//Applied Predictive Modelling//RNA_seq_1.xlsx')
    #Do basic preprocessing
    data=preprocessing_gene(data)
    #Read in clinical data
    clinical=pd.read_excel('C://Personal//Unf Course Folder//Applied Predictive Modelling//clinical.xlsx',na_values='nan')
    #do basic preprocessing
    clinical_data=preprocessing_clinical(clinical)
     #delete the overall survival column
    clinical_data.drop(['overallsurvival'],axis=1,inplace=True) 
    clinical_data=clinical_data.reset_index()
    clinical_data.drop(columns=["index"],inplace=True)
    #delete redundatnt rows
    for i in clinical_data['attrib_name']:
        if i not in data.index:
            clinical_data.drop(clinical_data[clinical_data['attrib_name']==i].index,inplace=True)
    clinical_data=clinical_data.set_index('attrib_name')   
    d=clinical_data.describe()
    #convert to integer and float
    for i in ['years_to_birth','Tumor_purity','overall_survival','status']:
        clinical_data[i]=clinical_data[i].astype('float')   
    for i in ['histological_type','gender','radiation_therapy','race','ethnicity']    :
        clinical_data[i]=clinical_data[i].astype('string') 
    #percentage of missing value 
    missing_perc={}
    features=['histological_type','gender','radiation_therapy','race','ethnicity']
    for i in features:
        missing_perc[i]=100*sum(clinical_data[i]=="nan")/clinical_data[i].size
    for i in ['years_to_birth','Tumor_purity','overall_survival','status']:
        missing_perc[i]=100*clinical_data[i].isnull().sum()/clinical_data[i].size
    #find out index number of the gene expression that are missing in the clinical data
    #indexlist=finder(data.index,clinical_data["attrib_name"])
    #now drop those row from our gene expression dataNo documentation available 
    data=data.drop(index="TCGA.16.1048")
    data=data.drop(index="TCGA.28.2501")
    data=data.drop(index="TCGA.28.2510")
    #Now both clinical and gene expression data has 525 rows
    #convert dataframe to numpy array
    X=np.asarray(data).astype(np.float32)
    #convert to train,test
    X_train, X_test, y_train, y_test = train_test_split(X, X, test_size=0.10, random_state=42)
    #define autoencoder parameter
    layer_sizes=[4570,2000,500,100,10]
    activations1=['sigmoid','sigmoid','sigmoid','sigmoid']
    activations2=['sigmoid','sigmoid','sigmoid','sigmoid']
    noise=True
    ac=Autoencoder(layer_sizes,activations1,activations2,noise)
    ac.set_noise(0.25)
    encoder=ac.encoder()
    decoder=ac.decoder()
    model=ac.create_model("mse",optimizer=SGD,lr=0.1)
    #print(encoder.summary())
    history=model.fit(X_train,y_train,epochs=100,batch_size=16,validation_split=0.15)
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.legend(['Training_loss','Validation_loss'])
    plt.show()
    columns=[]
    for i in range(1,11):
        columns.append("features"+str(i))
    predictions=pd.DataFrame(encoder.predict(X) ,columns=columns)
    #standerdize the features
    scalar=StandardScaler().fit(predictions)
    scaled=pd.DataFrame((scalar.fit_transform(predictions)))
    scaled.columns=columns

   # clinical_data.to_csv('out.csv', index=False)
    scaled.index=clinical_data.index
    finaldata=pd.concat([clinical_data,scaled],axis=1,join='inner')
    finaldata.to_csv("C://Personal//Unf Course Folder//Spring 2021//Applied Predictive Modelling//Feature_data//raw_all_denoised.csv", index=False)
    #deal with missing values
    finaldata['Tumor_purity'].replace(to_replace = np.nan, value =np.mean(finaldata['Tumor_purity']),inplace=True)
    msno.matrix(finaldata)
    finaldata=finaldata.dropna()
    #exploratory analysis
    categorical=['histological_type','gender','radiation_therapy','ethnicity','overall_survival']
    final_cat=finaldata.loc[:,categorical]
    sns.distplot(finaldata[finaldata['radiation_therapy']=="no"]['overall_survival'],kde=False,color='r')
    sns.distplot(finaldata[finaldata['radiation_therapy']=="yes"]['overall_survival'],kde=False,color='b')
    plt.show()
    sns.distplot(finaldata[finaldata['histological_type']=="untreatedprimary(denovo)gbm"]['overall_survival'],kde=False,color='r')
    sns.distplot(finaldata[finaldata['histological_type']=="treatedprimarygbm"]['overall_survival'],kde=False,color='b')
    sns.distplot(finaldata[finaldata['histological_type']=="glioblastomamultiforme(gbm)"]['overall_survival'],kde=False,color='g')
    plt.show()
    for i in ['histological_type','radiation_therapy','race','ethnicity']:
        sns.catplot(x=i,y="overall_survival",data=finaldata)
        plt.savefig('output{}.png'.format(i))
    #baseline model
    categorical=['histological_type','radiation_therapy','gender','race','ethnicity']
    for feature in categorical:
        le = LabelEncoder()
        finaldata[feature] = le.fit_transform(finaldata[feature].astype(str))
    finaldata.to_csv("C://Personal//Unf Course Folder//Spring 2021//Applied Predictive Modelling//Feature_data//raw_all_denoised_10.csv", index=False)
    #explore numerical features
    num=['years_to_birth','Tumor_purity','overall_survival','feature1','feature2','feature3','feature4','feature5']
    num_data=finaldata.loc[:,num]
    cor=num_data.corr()
    sns.boxplot(x=finaldata['overall_survival'])
    plt.savefig('overallsurvivalboxplot.png')
    sns.boxplot(x=finaldata['years_to_birth'])
    plt.savefig('years_to_birth.png')
    sns.scatterplot(x="years_to_birth",y="overall_survival",data=finaldata)
    plt.savefig('yearsvssurvival.png')
    #standerdization
    #scaler = StandardScaler()
    #scaled_finald=scaler.fit_transform(finaldata)
    #pd.DataFrame(scaled_finald).to_csv("C://Personal//Unf Course Folder//Spring 2021//Applied Predictive Modelling//Feature_data//scaled_all.csv", index=False)
    #train test split
    data=pd.read_csv('C://Personal//Unf Course Folder//Spring 2021//Applied Predictive Modelling//Feature_data//raw_all_denoised_10.csv')
    x_train,x_test,y_train,y_test=train_test_split(data,data,test_size=0.20,random_state=123)
    aft=WeibullAFTFitter()
    x_train=pd.DataFrame(x_train)
    x_test=pd.DataFrame(x_test)
    x_train.columns=finaldata.columns
    x_test.columns=finaldata.columns

    aft=aft.fit(pd.DataFrame(x_train), duration_col="overall_survival", event_col="status")
    p=aft.predict_expectation(x_test)
    c=concordance_index(x_test["overall_survival"],p,x_test["status"])
    #select only clinical features    
    data=pd.read_csv('C://Personal//Unf Course Folder//Spring 2021//Applied Predictive Modelling//Feature_data//raw_all_encoded.csv')
    data.drop(['feature1','feature2','feature3','feature4','feature5'],axis=1,inplace=True)
    x_train,x_test,y_train,y_test=train_test_split(data,data,test_size=0.20,random_state=123)
    aft=WeibullAFTFitter()
    x_train=pd.DataFrame(x_train)
    x_test=pd.DataFrame(x_test)
    x_train.columns=data.columns
    x_test.columns=data.columns

    aft=aft.fit(pd.DataFrame(x_train), duration_col="overall_survival", event_col="status")
    p=aft.predict_expectation(x_test)
    c=concordance_index(x_test["overall_survival"],p,x_test["status"])
    
    
    ###
    final_samples=finaldata["attrib_name"]
    finaldata.drop(columns=["histological_type","radiation_therapy","race","ethnicity",],inplace=True)
    finaldata.columns=['years_to_birth','Tumor_purity','gender','overall_survival','status','feature1','feature2','feature3','feature4','feature5']
    finaldata["years_to_birth"]=finaldata["years_to_birth"].astype(int)
    finaldata["Tumor_purity"]=finaldata["Tumor_purity"].astype(float)
    finaldata=finaldata[finaldata["status"]!="nan"]
    finaldata["status"]=finaldata["status"].astype(int)
    lc=LabelEncoder()
    finaldata["gender"]=lc.fit_transform(finaldata.gender)
    #Replace Nan Tumor purity mean values
    
    finaldata.replace(to_replace = np.nan, value =0,inplace=True)
    finaldata['Tumor_purity'].replace(to_replace = np.nan, value =0.8456692607003896,inplace=True)
    #finaldata.drop(columns=["feature1","feature2","feature3","feature4","feature5"],inplace=True)
    
    finaldata['overall_survival']=finaldata['overall_survival'].astype('float')
    num=['years_to_birth','Tumor_purity','overall_survival','feature1','feature2','feature3','feature4','feature5']
    num_data=finaldata.loc[:,num]
    cor=num_data.corr()
    print(finaldata.groupby("gender")["overall_survival"].mean())

    for i in ['years_to_birth','Tumor_purity','overall_survival','status']:
        clinical_data[i]=clinical_data[i].astype('float')
    print(clinical_data.groupby('histological_type')['overall_survival'].mean()) 
    print(clinical_data.groupby('gender')['overall_survival'].mean()) 
    print(clinical_data.groupby('radiation_therapy')['overall_survival'].mean())
    
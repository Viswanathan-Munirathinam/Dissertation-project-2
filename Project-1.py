#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# In[2]:


df= pd.read_excel("C:/Users/Viswanathan/Desktop/Gestational Diabetic Dat Set.xlsx")


# In[3]:


df.head()


# In[5]:


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer = imputer.fit(df[["BMI"]])
df["BMI"] = imputer.transform(df[["BMI"]])
imputer=imputer.fit(df[["HDL"]])
df["HDL"]=imputer.transform(df[["HDL"]])
imputer=imputer.fit(df[["Sys BP"]])
df["Sys BP"]=imputer.transform(df[["Sys BP"]])
imputer=imputer.fit(df[["OGTT"]])
df["OGTT"]=imputer.transform(df[["OGTT"]])


# In[10]:


df.isnull().sum()


# In[86]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVR
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression


# In[39]:


X=df[["Age","No of Pregnancy","Gestation in previous Pregnancy","BMI","HDL","Family History","unexplained prenetal loss","Large Child or Birth Default",
"PCOS","Sys BP","Dia BP","OGTT", "Hemoglobin", "Sedentary Lifestyle", "Prediabetes"]]
y=df["Target"]


# In[37]:


estimator = SVR(kernel="linear")

selector = RFE(estimator, n_features_to_select=5, step=1)

selector.fit(X, y)


# In[38]:


print(selector.support_)


# In[40]:


print(selector.ranking_)


# In[ ]:





# In[27]:


pip install feature_engine


# In[84]:


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.feature_selection import RFE

X=df[["Age","No of Pregnancy","Gestation in previous Pregnancy","BMI","HDL","Family History","unexplained prenetal loss","Large Child or Birth Default",
"PCOS","Sys BP","Dia BP","OGTT", "Hemoglobin", "Sedentary Lifestyle", "Prediabetes"]]
y=df["Target"]
X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=42, test_size= 0.20, shuffle=True)

svc = SVC(kernel="linear", C=5)
rfe = RFE(estimator=svc, n_features_to_select=8, step=1)
rfe.fit(X, y)


# In[79]:


ranking = rfe.ranking_
print(ranking)


# In[57]:


CorrMat=df.corr()
plt.figure(figsize=(15,20))
sns.heatmap(CorrMat,annot=True)


# In[75]:


df1=df[["Age","Gestation in previous Pregnancy","BMI","PCOS","Dia BP","OGTT", "Hemoglobin", "Prediabetes", "Target"]]


# In[76]:


df1.head()


# In[77]:


CorrMat=df1.corr()
plt.figure(figsize=(15,20))
sns.heatmap(CorrMat,annot=True)


# In[98]:


X=df1[["Age","Gestation in previous Pregnancy","BMI",
"PCOS","Dia BP","OGTT", "Hemoglobin", "Prediabetes"]]
y=df1["Target"]
X_train, X_test, y_train, y_test =train_test_split(X,y,test_size=0.20,shuffle=True,random_state=8)


# In[99]:


X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, 
    test_size=0.25, random_state= 8) 


# In[100]:


print("X_train shape: {}".format(X_train.shape))
print("X_test shape: {}".format(X_test.shape))
print("y_train shape: {}".format(y_train.shape))
print("y_test shape: {}".format(y_test.shape))
print("X_val shape: {}".format(y_train.shape))
print("y val shape: {}".format(y_test.shape))


# In[ ]:





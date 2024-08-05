#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# In[3]:


df= pd.read_excel("C:/Users/Viswanathan/Desktop/Gestational Diabetic Dat Set.xlsx")


# In[92]:


df.dtypes


# In[ ]:


df.isnull()


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


# In[6]:


df.isnull().sum()


# In[7]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVR
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression


# In[8]:


X=df[["Age","No of Pregnancy","Gestation in previous Pregnancy","BMI","HDL","Family History","unexplained prenetal loss","Large Child or Birth Default",
"PCOS","Sys BP","Dia BP","OGTT", "Hemoglobin", "Sedentary Lifestyle", "Prediabetes"]]
y=df["Target"]


# In[9]:


estimator = SVR(kernel="linear")

selector = RFE(estimator, n_features_to_select=5, step=1)

selector.fit(X, y)


# In[10]:


print(selector.support_)


# In[11]:


print(selector.ranking_)


# In[12]:


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


# In[13]:


ranking = rfe.ranking_
print(ranking)


# In[37]:


CorrMat=df.corr()
plt.figure(figsize=(15,20))
sns.heatmap(CorrMat,annot=True)


# In[15]:


df1=df[["Age","Gestation in previous Pregnancy","BMI","PCOS","Dia BP","OGTT", "Hemoglobin", "Prediabetes", "Target"]]


# In[16]:


df1.head()


# In[17]:


CorrMat=df1.corr()
plt.figure(figsize=(15,20))
sns.heatmap(CorrMat,annot=True)


# In[18]:


X=df1[["Age","Gestation in previous Pregnancy","BMI",
"PCOS","Dia BP","OGTT", "Hemoglobin", "Prediabetes"]]
y=df1["Target"]
X_train, X_test, y_train, y_test =train_test_split(X,y,test_size=0.20,shuffle=True,random_state=8)


# In[19]:


X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, 
    test_size=0.25, random_state= 8) 


# In[20]:


print("X_train shape: {}".format(X_train.shape))
print("X_test shape: {}".format(X_test.shape))
print("y_train shape: {}".format(y_train.shape))
print("y_test shape: {}".format(y_test.shape))
print("X_val shape: {}".format(y_train.shape))
print("y val shape: {}".format(y_test.shape))


# In[17]:


from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=200)
rfc.fit(X_train, y_train)


# In[18]:


rfc_train = rfc.predict(X_train)
from sklearn import metrics
print("Training Accuracy =", format(metrics.accuracy_score(y_train, rfc_train)))


# In[85]:


rfc_predictions = rfc.predict(X_test)
print("Test Accuracy =", format(metrics.accuracy_score(y_test, rfc_predictions)))


# In[76]:


rfc.feature_importances_
pd.Series(rfc.feature_importances_, index=X.columns).plot(kind='barh')


# In[20]:


from sklearn.tree import DecisionTreeClassifier

dtree = DecisionTreeClassifier()
dtree.fit(X_train, y_train)


# In[82]:


dtree_predictions = dtree.predict(X_test)
print("Test Accuracy =", format(metrics.accuracy_score(y_test,dtree_predictions)))


# In[33]:


from sklearn.svm import SVC

svc_model = SVC()
svc_model.fit(X_train, y_train)


# In[34]:


svc_pred = svc_model.predict(X_test)
print("Test Accuracy =", format(metrics.accuracy_score(y_test, svc_pred)))


# In[27]:


dtree.feature_importances_


# In[28]:


pd.Series(dtree.feature_importances_, index=X.columns).plot(kind='barh')


# In[88]:


from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
model = GaussianNB()
model.fit(X_train, y_train)
Naivebayes_pred =model.predict(X_test)


# In[89]:


print ("accuracy:", metrics.accuracy_score (y_test, Naivebayes_pred))


# In[53]:


from sklearn.neighbors import KNeighborsClassifier


# In[55]:


knn = KNeighborsClassifier()

knn.fit(X_train,y_train)
knn.score(X_test,y_test)


# In[57]:


from sklearn.model_selection import KFold, cross_val_score


# In[59]:


clf=DecisionTreeClassifier(random_state=42)
k_folds=KFold(n_splits=10)
scores=cross_val_score(clf,X_train,y_train,cv=k_folds)


# In[60]:


print("Cross Validation Scores: ", scores)
print("Average CV Score: ", scores.mean())
print("Number of CV Scores used in Average: ", len(scores))


# In[79]:


# To find the performance of the model calculated accuracy.
from sklearn.metrics import classification_report, confusion_matrix

print(confusion_matrix(y_test, svc_pred))
print(classification_report(y_test, svc_pred))


# In[83]:


from sklearn.metrics import classification_report, confusion_matrix

print(confusion_matrix(y_test, dtree_predictions))
print(classification_report(y_test, dtree_predictions))


# In[86]:


from sklearn.metrics import classification_report, confusion_matrix

print(confusion_matrix(y_test, rfc_predictions))
print(classification_report(y_test, rfc_predictions))


# In[90]:


from sklearn.metrics import classification_report, confusion_matrix

print(confusion_matrix(y_test,Naivebayes_pred))
print(classification_report(y_test,Naivebayes_pred))


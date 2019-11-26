#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.metrics import mean_absolute_error


# In[ ]:


ankitaTrain = pd.read_csv('tcd-ml-1920-group-income-train.csv')
ankitaTest = pd.read_csv('tcd-ml-1920-group-income-test.csv')    


# In[ ]:


ankitaTrain.head()


# In[ ]:


ankitaTest.head()


# In[ ]:


ankitaTrain[ankitaTrain.columns]=ankitaTrain[ankitaTrain.columns].fillna(ankitaTrain.mode().iloc[0])
ankitaTest[ankitaTest.columns]=ankitaTest[ankitaTest.columns].fillna(ankitaTest.mode().iloc[0])


# In[ ]:


ankitaTrain['Gender'].value_counts()


# In[ ]:


ankitaTrain["Gender"] = ankitaTrain["Gender"].replace('f','female')
ankitaTrain["Gender"] = ankitaTrain["Gender"].replace('0','unknown')


# In[ ]:


ankitaTrain['Gender'].value_counts()


# In[ ]:


ankitaTest['Gender'].value_counts()


# In[ ]:


ankitaTest["Gender"] = ankitaTest["Gender"].replace('f','female')
ankitaTest["Gender"] = ankitaTest["Gender"].replace('0','unknown')


# In[ ]:


a = statistics.mode(ankitaTrain['Housing Situation'])
ankitaTrain['Housing Situation'] = ankitaTrain['Housing Situation'].replace('nA',a)
ankitaTrain['Housing Situation'] = ankitaTrain['Housing Situation'].astype(str)


# In[ ]:


a = statistics.mode(ankitaTest['Housing Situation'])
ankitaTest['Housing Situation'] = ankitaTest['Housing Situation'].replace('nA',a)
ankitaTest['Housing Situation'] = ankitaTest['Housing Situation'].astype(str)


# In[ ]:


ankitaTest['University Degree'] = ankitaTest['University Degree'].replace('0','No')
ankitaTrain['University Degree'] = ankitaTrain['University Degree'].replace('0','No')


# In[ ]:


ankitaTrain['Work Experience in Current Job [years]'] = ankitaTrain['Work Experience in Current Job [years]'].replace('#NUM!',ankitaTrain['Work Experience in Current Job [years]'].mode())
ankitaTest['Work Experience in Current Job [years]'] = ankitaTest['Work Experience in Current Job [years]'].replace('#NUM!',ankitaTest['Work Experience in Current Job [years]'].mode())


# In[ ]:


data = pd.concat([ankitaTrain, ankitaTest], ignore_index=True)


# In[ ]:


data['Yearly Income in addition to Salary (e.g. Rental Income)'] = data['Yearly Income in addition to Salary (e.g. Rental Income)'].str.replace('([A-Za-z]+)', '')
data['Yearly Income in addition to Salary (e.g. Rental Income)'] = pd.to_numeric(data['Yearly Income in addition to Salary (e.g. Rental Income)'])


# In[ ]:


data['Work Experience in Current Job [years]'] = pd.to_numeric(data['Work Experience in Current Job [years]'])


# In[ ]:


data.dtypes


# In[ ]:


for i in data.dtypes[data.dtypes == 'object'].index.tolist():
    grp_hs = data.groupby(i)["Total Yearly Income [EUR]"].agg("mean")
#     data[i] = featureLE.transform(data[i].astype(str))
    data[i] = data[i].map(grp_hs) 


# In[ ]:


data.head()


# In[ ]:


data.columns


# In[ ]:


data.shape


# In[ ]:


ankitaTrain.shape


# In[ ]:


ankitaTest.shape


# In[ ]:


XTrain,XTest = data[data.columns].iloc[:1048573],data[data.columns].iloc[1048574:]


# In[ ]:


YTrain = data['Total Yearly Income [EUR]'].iloc[:1048573]


# In[ ]:


XTrain.drop(['Instance', 'Total Yearly Income [EUR]'], axis=1 , inplace=True)
XTest.drop(['Instance', 'Total Yearly Income [EUR]'], axis=1, inplace=True)


# In[ ]:


print(XTrain.shape)
print(XTest.shape)
print(YTrain.shape)


# In[ ]:


reg = LinearRegression().fit(XTrain, YTrain)


# In[ ]:


XTest[XTest.columns]=XTest[XTest.columns].fillna(XTest.mode().iloc[0])


# In[ ]:


xTrain,xValidation,yTrain,yValidation = train_test_split(XTrain,YTrain,test_size=0.2,random_state=1234)    


# In[ ]:


reg = LinearRegression().fit(xTrain, yTrain)


# In[ ]:


y_predict = reg.predict(XTest)


# In[ ]:


ak = pd.read_csv('tcd-ml-1920-group-income-submission.csv')


# In[ ]:


ak['Total Yearly Income [EUR]'] = y_predict


# In[ ]:


ak


# In[ ]:


ak.to_csv('ak.csv',index=False)


# In[ ]:


model=CatBoostRegressor(iterations=2000, depth=10, learning_rate=0.01, loss_function='MAE')
model.fit(xTrain,yTrain,cat_features=None,eval_set=(xValidation, yValidation))
y_predi = model.predict(xValidation)


# In[ ]:


y_predi = model.predict(xValidation)
mean_absolute_error(yValidation, y_predi)


# In[ ]:


y_predict = model.predict(XTest)
ak = pd.read_csv('tcd-ml-1920-group-income-submission.csv')
ak['Total Yearly Income [EUR]'] = y_predict
ak.to_csv('MLSub.csv',index=False)


# In[ ]:


ak


# In[ ]:





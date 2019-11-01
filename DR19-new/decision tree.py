# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 14:11:48 2019

@author: ecs
"""
import pandas as pd  
import numpy as np  
from sklearn.tree import DecisionTreeClassifier


data = pd.read_excel('Datasets/Gear1.xlsx')
# Dataset is now stored in a Pandas Dataframe
data = data.dropna(axis = 1, how ='all') 

#removing columns havin more than 90% zero
data=data.drop(columns=data.columns[((data==0).mean()>0.90)],axis=1)
print(data.info)

#saving lables in y
X_data= data.iloc[:,:-1].values
y_data=data.iloc[:,-1].values

#Normalization
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_data=sc.fit_transform(X_data)

#splitting data into train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2)
X_train, X_val, y_train, y_val   = train_test_split(X_train, y_train, test_size=0.2, random_state=1)
from sklearn.model_selection import cross_val_score
model = DecisionTreeClassifier(random_state=0)

model.fit(X_train,y_train)
y_pred = model.predict(X_test)
import sklearn
from sklearn.model_selection import cross_val_score
Accuracy = cross_val_score(model, X_train, y_train, cv=10 , scoring="accuracy")
f1_score=cross_val_score(model, X_train, y_train, cv=10, scoring='f1_macro')
print('decision tree')

print (Accuracy)
print (f1_score)
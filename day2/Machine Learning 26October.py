#!/usr/bin/env python
# coding: utf-8

# # Basic Machine Learninh ALgorithms and Their Applications - Exercise

# Make sure that Python 2 or 3 is installed before you proceed. With this exercise.
# 
# We will be using scikit-learn library. If you don’t have scikit-learn installed yet, run <b> pip install scikit-learn </b> in your terminal. That should handle installing scikit-learn and all prerequisites libraries that we will need.

# In[1]:


#importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# We will use Wisconsin Breast Cancer dataset which records clinical measurements of breast cancer tumors. Each tumor is labeled as “benign” (for harmless tumors) or “malignant” (for cancerous tumors), and the task is to learn to predict whether a tumor is malignant based on the measurements of the tissue.

# The dataset is available in scikit learn and can be loaded using details available in sklearn documentation https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html .However, we will use the same dataset that is available in Kaggle (https://www.kaggle.com/ ).

# <h3> Loading the Data set </h3>

# In[2]:


'''the dataset should be in the same folder as python file you are working on for the following command to work. '''
df = pd.read_csv('cancer_data.csv')


# In[ ]:


#Display part of the imported dataset
df


# The following commands provide details of our dataset: Size, outputs available and features available

# In[ ]:


# counting values of variables in 'diagnosis'
df['diagnosis'].value_counts()


# In[ ]:


# visualize the count:
sns.countplot(df['diagnosis'], label = 'count')


# <h3>Preprocessing</h3>

# In[ ]:


#checking for null values in the dataset
df.isnull().sum()


# In[7]:


#droping the feature (Unnamed:32)
df.drop(['Unnamed: 32','id'],axis=1,inplace=True)


# Divide the dataset into independent variables (predictors) and target

# In[8]:


#predictors
x= df.drop('diagnosis',axis=1)
#target
y = df.diagnosis


# Our target contains categorical data and we have to convert categorical data into the binary format for further process. We use Scikit learn Label Encoder for encoding the categorical data.

# In[9]:


from sklearn.preprocessing import LabelEncoder
#creating the object
lb = LabelEncoder()
y = lb.fit_transform(y)


# We split the data into training and testing datasets

# In[10]:


from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.3,random_state=40)


# In an artificial neural network, weights and input data of nodes are multiplied therefore taking a lot of time. We scale the data into smaller numbers to reduce the time. <b> StandardScaler </b> module from scikit learn is used.

# In[11]:


#importing StandardScaler
from sklearn.preprocessing import StandardScaler
#creating object
sc = StandardScaler()
xtrain = sc.fit_transform(xtrain)
xtest = sc.transform(xtest)


# <h3>Fitting the Model</h3>

# In[16]:


from sklearn.neural_network import MLPClassifier


# In[17]:


clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)


# In[ ]:


clf.fit(xtrain,ytrain)


# In[ ]:


clf.score(xtest,ytest)


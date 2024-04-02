#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.cm import rainbow
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')


# In[2]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# In[11]:


df= pd.read_csv ('heart_disease.csv')


# In[7]:


import os


# In[8]:


os.getcwd()


# In[10]:


os.chdir('C:\\Users\\Ayush\\Downloads')


# In[12]:


df.info()


# In[13]:


df.describe()


# In[14]:


import seaborn as sns
#get correlations of each features in dataset
corrmat = df.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20,20))
#plot heat map
g=sns.heatmap(df[top_corr_features].corr(),annot=True,cmap="RdYlGn")


# In[15]:


df.hist()


# In[16]:


sns.set_style('whitegrid')
sns.countplot(x='target',data=df,palette='RdBu_r')


# In[17]:


dataset = pd.get_dummies(df, columns = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'])


# In[18]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
standardScaler = StandardScaler()
columns_to_scale = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
dataset[columns_to_scale] = standardScaler.fit_transform(dataset[columns_to_scale])


# In[19]:


dataset.head()


# In[20]:


y = dataset['target']
X = dataset.drop(['target'], axis = 1)


# In[29]:


from sklearn.model_selection import cross_val_score
knn_scores = []
for k in range(1,32):
    knn_classifier = KNeighborsClassifier(n_neighbors = k)
    score=cross_val_score(knn_classifier,X,y,cv=10)
    knn_scores.append(score.mean())


# In[30]:


plt.plot([k for k in range(1, 32)], knn_scores, color = 'red')
for i in range(1,32):
    plt.text(i, knn_scores[i-1], (i, knn_scores[i-1]))
plt.xticks([i for i in range(1, 32)])
plt.xlabel('Number of Neighbors (K)')
plt.ylabel('Scores')
plt.title('K Neighbors Classifier scores for different K values')


# In[31]:


knn_classifier = KNeighborsClassifier(n_neighbors = 12)
score=cross_val_score(knn_classifier,X,y,cv=10)


# In[32]:


score.mean()


# In[33]:


from sklearn.ensemble import RandomForestClassifier


# In[34]:


randomforest_classifier= RandomForestClassifier(n_estimators=10)

score=cross_val_score(randomforest_classifier,X,y,cv=10)


# In[35]:


score.mean()


# In[ ]:





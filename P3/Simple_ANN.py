#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import tensorflow as tf


# In[3]:


x = np.linspace(0,4*np.pi,60000)
X = []
for i in range(len(x)):
    X.append([x[i]])
X[7500]


# In[4]:


Y = np.sin(x)
y = []
for i in range(len(Y)):
    y.append([Y[i]])


# In[5]:


from sklearn.preprocessing import StandardScaler


# In[6]:


sc = StandardScaler()


# In[7]:


X = sc.fit_transform(X)
X


# In[8]:


y = sc.transform(y)
y


# In[9]:


from sklearn.model_selection import train_test_split


# In[10]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# In[11]:


len(y_train)


# In[12]:


len(y_test)


# In[13]:


ann = tf.keras.models.Sequential()


# In[14]:


ann.add(tf.keras.layers.Dense(units=6,activation='sigmoid'))


# In[15]:


ann.add(tf.keras.layers.Dense(units=6,activation='sigmoid'))


# In[16]:


ann.add(tf.keras.layers.Dense(units=1,activation='sigmoid'))


# In[17]:


ann.compile(optimizer = 'adam' , loss = 'mean_squared_error' , metrics = ['mean_squared_error'])


# In[18]:


ann.fit(X_train,y_train,batch_size=32,epochs=100)


# In[22]:


o = [0.00001]
o


# In[23]:


o = sc.transform([o]) 
o


# In[24]:


ann.predict(o)


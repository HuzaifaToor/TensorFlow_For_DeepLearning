#!/usr/bin/env python
# coding: utf-8

# In[2]:


import tensorflow as tf
# Regression Modeling


# In[48]:


import numpy as np
import matplotlib.pyplot as plt 
#x = np.arange(-20,20,3)
#y = np.arange(20,60,3)
x = np.arange(-100,100,2)
y = (x*2)+5
#y = np.arange(-100,100,20)
x = tf.constant(x)
y = tf.constant(y)
plt.scatter(x,y)


# In[49]:


#relation pridiction b/w x and y
y == (x*2)+5


# In[10]:


# Creating demo tensor for housing problem

house_info = tf.constant(["Bedroom", "Bathrom", "garage"])
house_price = tf.constant([939700])

print(house_info)
print(house_price)


# In[3]:


print(x.shape)
print(y.shape)


# In[4]:


#Steps in Modeling with TensorFlow   
#1) Creating Mode 2) Compiling a Model 3) Fitting a Model

tf.random.set_seed(42)

#1) Creating Model using sequential API
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1)
])
#Another method
#model = tf.keras.Sequential()
#model.add(tf.keras.layers.Dense(1))

#########################
#2) Compiling a Model
# mae = mean absolute error
# SGD = stochastic gradient decent (Google it)

model.compile(loss=tf.keras.losses.mae,
             optimizer=tf.keras.optimizers.SGD(),
             metrics=["mae"])

#check by writing "tf.keras.losses.mae" on google
###################################################
#3) Fitting a Model
#(epoch means no of oppertunities for system to go through inputs and outputs to improve prediction)
model.fit(x,y,epochs=1000, verbose = 0) #high epoch = high accuracy 


# In[12]:


#make pridiction on previous model training
print(model.predict([-10.0]))


# In[54]:


#Improving the Model
# add more layers
#Increase no of hidden units
#Can change optimizer function in compilation or learning   

#rebuild model
#defining
model = tf.keras.Sequential([
    tf.keras.layers.Dense(100, activation=None),
    #tf.keras.layers.Dense(100, activation="relu"),
    #tf.keras.layers.Dense(100, activation="relu"),
    tf.keras.layers.Dense(1)
])

#compiling
model.compile(loss="mae",
             optimizer=tf.keras.optimizers.Adam(learning_rate = 0.01),
             metrics=["mae"])

#fitting the model
model.fit(x,y,epochs=1000, verbose = 0)


# In[55]:


print(model.predict([110.0]))
#Hence model is improved by adjusting sequential hidden layers,
#optimizer and learning rate, epoc (can change activation func as well)


# In[57]:


#In practicel machine learning problems, we usually have 3 sets 
#of data
#1) Training set (Model learns from this, usually 70-80%
#of the avaialbe data)
#2) Validation dataset (On which model performance is tuned and judged)
#3) Test set (Model gets evaluated on this (10-15% of total))

#Now splitting our data into train and test data sets

x_train = x[:80]
y_train = y[:80]

x_test = x[80:]
y_test = y[80:]
print(len(x_train), len(y_train), len(x_test), len(y_test))


# In[60]:


#Plotting to visulize

plt.figure(figsize = (10,7))
plt.scatter(x_train,y_train, c="g", label = "Training Data")
plt.scatter(x_test,y_test, c="r", label = "Testing Data")
plt.legend()


# In[63]:


#Training model on the basis of training sets


#defining
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation=None),
    #tf.keras.layers.Dense(100, activation="relu"),
    #tf.keras.layers.Dense(100, activation="relu"),
    tf.keras.layers.Dense(1)
])

#compiling
model.compile(loss="mae",
             optimizer=tf.keras.optimizers.Adam(learning_rate = 0.01),
             metrics=["mae"])

#fitting the model
model.fit(x_train,y_train,epochs=100, verbose = 0)
# verbose = 0 will make output not to show up

#below command shows the info about model parameters
model.summary()

#pictorial dipiction
from tensorflow.keras.utils import plot_model

plot_model(model = model)


# In[67]:



# In[ ]:





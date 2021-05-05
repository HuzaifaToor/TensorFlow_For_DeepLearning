#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
print(tf.__version__)


# In[4]:


#create tensor with tf.contant (Values can't be changed)
scalar = tf.constant(8)
print(scalar)


# In[5]:


#no of dimentions
print(scalar.ndim)


# In[6]:


vector = tf.constant([6,6])
print(vector)
print(vector.ndim)


# In[9]:


#matrix_2D
matrix = tf.constant([[8,5],[9,7]])
print(matrix)
print(matrix.ndim)


# In[12]:


#matrix_3D
matrix = tf.constant([[[3,8],[5,6]],[[8,5],[9,7]]], dtype = tf.float32)
print(matrix)
print(matrix.ndim)


# In[24]:


#create tensor with tf.variable (Values can be changed)
changeable_tensor = tf.Variable([10,7])
print(changeable_tensor)
changeable_tensor[0].assign(99)
print(changeable_tensor)


# In[27]:


#create random tensors
random1= tf.random.Generator.from_seed(42) #set seed for reproducibility
random1=random1.normal(shape=(4, 3, 2))
print(random1)
#Checking if ranom matrices generated from same seed are equal
random2= tf.random.Generator.from_seed(42)
random2=random2.normal(shape=(4, 3, 2))
print(random1, random2, random1 == random2)


# In[33]:


#Shuffeling Tensors

unshuffled = tf.constant([[6,5],[10,6],[9,4]])
print(unshuffled)
print(tf.random.shuffle(unshuffled))


# In[43]:





# In[42]:


#Same As numpy, tf.zeroes() and tf.ones() can be created
#Tensor can be run on GPU much faster while numpy arrays can't
import numpy as np

array1 = np.arange(1,25, dtype = np.int32)
print(array1)
tfarray = tf.constant(array1, shape = (2,3,4))
print(tfarray)


# In[ ]:





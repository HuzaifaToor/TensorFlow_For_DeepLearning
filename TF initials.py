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


# In[16]:


#create tensor with tf.variable (Values can be changed)
changeable_tensor = tf.Variable([10,7])
print(changeable_tensor)
changeable_tensor[0].assign(99)
print(changeable_tensor)


# In[21]:


#create random tensors
random1= tf.random.Generator.from_seed(42) #set seed for reproducibility
random1=random1.normal(shape=(4, 3, 2))
print(random1)


# In[ ]:





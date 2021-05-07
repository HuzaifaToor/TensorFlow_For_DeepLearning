#!/usr/bin/env python
# coding: utf-8

# In[2]:


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


# In[4]:


#shape size dimention aix etc of tensors

rank4D_tensor = tf.ones(shape = [2,3,4,5])
print(rank4D_tensor)


# In[6]:


# getting specific elements of tensor
print(rank4D_tensor[:2, :2, :2, :2])
print(rank4D_tensor[:3, :2, :2, :3])


# In[7]:


print(rank4D_tensor[:2, :2, :2])


# In[8]:


print(rank4D_tensor[:2, :2, :, :2])


# In[14]:


# Adding rank or dimention
rank2tensor = tf.constant([[2,5],[9,7]])
print(rank2tensor)
#print last column
print(rank2tensor[:, -1])
#Add rank
rank3tensor= rank2tensor[..., tf.newaxis]
print(rank3tensor)
#another method
print(tf.expand_dims(rank2tensor, axis = -1)) # -1 for last axis


# In[25]:


#matrix multiplication tensorflow syntax
print(tf.matmul(rank2tensor,rank2tensor))
print(tf.tensordot(rank2tensor,rank2tensor, axes = 0))
#simple python matrix multiplication syntax
print(rank2tensor@rank2tensor)
#transpose
print(tf.transpose(rank2tensor))
#reshapping tesnor
print(tf.reshape(rank2tensor, shape=(1,4)))


# In[46]:


#changing datatypes
import numpy as np
tensor1 = tf.constant([[-1,2],[3,-4]])
print(tensor1)
tensor2 = tf.cast(tensor1, dtype = tf.float16)
print(tensor2)
#absolute
print(tf.abs(tensor2))
#Find min max mean sum
a = tf.constant(np.random.randint(0,100, size=20))
print(a)
print(tf.reduce_min(a))
print(tf.reduce_max(a))
print(tf.reduce_mean(a))
print(tf.reduce_sum(a))
print(tf.math.reduce_std(tf.cast(a, dtype = tf.float32))) #need type cast to float
print(tf.math.reduce_variance(tf.cast(a, dtype = tf.float32)))
print(np.std(a))
print(np.var(a))


# In[51]:


#positional max and min
tf.random.set_seed(42)
b = tf.random.uniform(shape=[50])
print(b)
print(tf.argmax(b))
print(b[tf.argmax(b)])
print(b[tf.argmin(b)])


# In[56]:


# sequeezing a tensor
tf.random.set_seed(42)
c =tf.constant(tf.random.uniform(shape=[50]), shape = (1,1,1,1,50))
print(c)
print(tf.squeeze(c))


# In[59]:


tf.config.list_physical_devices("CPU")


# In[ ]:





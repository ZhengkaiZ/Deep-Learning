
# coding: utf-8

# In[1]:


import tensorflow as tf


# In[2]:


print(tf.__version__)


# In[3]:


hello = tf.constant("hello")


# In[6]:


type(hello)


# In[11]:


with tf.Session() as sess:
    output = sess.run(hello)


# In[12]:


print(output)


# In[13]:


type(output)


# In[14]:


tensor_1 = tf.constant(1)


# In[15]:


tensor_2 = tf.constant(2)


# In[16]:


tensor_sum = tensor_1 + tensor_2


# In[17]:


sess


# In[20]:


with tf.Session() as sess:
    output = sess.run(tensor_sum)


# In[22]:


print(output)


# In[24]:


print(tf.get_default_graph())


# In[25]:


g = tf.Graph()


# In[26]:


print(g)


# In[28]:


with g.as_default():
    print(g is tf.get_default_graph())


# In[29]:


print(g is tf.get_default_graph())


# In[32]:


sess = tf.InteractiveSession()


# In[35]:


my_tensor = tf.random_uniform(shape=(4,4), minval=0, maxval=10)


# In[36]:


print(my_tensor)


# In[43]:


tensor_2 = tf.constant([1,2,3,4])


# In[44]:


my_var = tf.Variable(initial_value=my_tensor)


# In[ ]:





# In[ ]:





# In[49]:


my_var.eval()


# In[50]:


type(my_var)


# In[51]:


type(tensor_2)


# In[52]:


my_var_2 = tf.Variable(initial_value=tensor_2)


# In[55]:


init = tf.global_variables_initializer()


# In[56]:


init.run()


# In[57]:


my_var_2.eval()


# In[59]:


ph = tf.placeholder(dtype=tf.float64)


# In[61]:


ph = tf.placeholder(dtype=tf.int32)


# In[62]:


ph = tf.placeholder(dtype=tf.int32, shape=(None, 5))


# In[ ]:






# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# In[42]:


x_data = np.linspace(0.0, 10.0, 1000000)
noise = np.random.random(size=len(x_data))
y_ture = x_data * 0.5 + 5 + noise

# Panda concat data
my_data = pd.concat([pd.DataFrame(data=x_data,columns=['x']), pd.DataFrame(data=y_ture,columns=['y'])],axis=1)
#my_data.head()
#my_data.plot(kind='scatter',x='x',y='y')
batch_size = 8

m = tf.Variable(0.5)
b = tf.Variable(5.0)

xph = tf.placeholder(tf.float32,[batch_size])
yph = tf.placeholder(tf.float32,[batch_size])
y_model = m * xph + b

error = tf.reduce_sum(tf.square(yph - y_model))
# Optimizer 
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0001)
train = optimizer.minimize(error)

init = tf.global_variables_initializer();

with tf.Session() as sess:
    sess.run(init)
    
    epoches = 1000
    for i in range(epoches):
        rand_ind = np.random.randint(len(x_data), size = batch_size)
        feed = {xph:x_data[rand_ind], yph:y_ture[rand_ind]}
        sess.run(train, feed_dict=feed)
    print(rand_ind)
    model_m,model_b = sess.run([m, b])

y_pred = x_data * model_m + model_b
#my_data.sample(200).plot(kind='scatter', x='x', y='y')
#plt.plot(x_data, y_pred, 'r')


# In[60]:


# Estimator API
from sklearn.model_selection import train_test_split

feat_column = [tf.feature_column.numeric_column('x',shape=[1])]
estimator = tf.estimator.LinearRegressor(feature_columns=feat_column)

x_train, x_test, y_train, y_test = train_test_split(x_data, y_ture, test_size=0.3, random_state=101)

input_func = tf.estimator.inputs.numpy_input_fn({'x':x_train},y_train,batch_size=4,num_epochs=None,shuffle=True)
train_func = tf.estimator.inputs.numpy_input_fn({'x':x_train}, y_train, batch_size=4,num_epochs=None, shuffle=False)
test_func = tf.estimator.inputs.numpy_input_fn({'x':x_test},y_test, batch_size=4,num_epochs=None, shuffle=False)

estimator.train(input_fn=input_func,steps=1000)

train_metircs = estimator.evaluate(input_fn=train_func,steps=1000)
test_matrics = estimator.evaluate(input_fn=test_func,steps=1000)

input_pred_func = tf.estimator.inputs.numpy_input_fn({'x':np.linspace(0, 10, 10)}, shuffle=False)

pred_lists = estimator.predict(input_fn=input_pred_func)
prediction = []
for l in pred_lists:
    prediction.append(l['predictions'])

plt.plot(np.linspace(0, 10, 10), prediction, 'r')
    


# In[ ]:





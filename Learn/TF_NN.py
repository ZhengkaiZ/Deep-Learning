import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

features = 20
hidden_units = 5

x = tf.placeholder(dtype=tf.float32, shape=(None, features))
W = tf.Variable(initial_value = tf.random_normal(dtype=tf.float32, shape=(features, hidden_units)))
b = tf.Variable(initial_value = tf.random_uniform(dtype=tf.float32, shape=(1,hidden_units)))

xW = tf.matmul(x, W)

z = tf.add(xW, b)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    
    output = sess.run(z, feed_dict={x:np.random.random([3, features])})
    print (output)


### Full Network Exampled
x_data = np.linspace(0, 10, 10) + np.random.uniform(-1.5, 1.5, 10)
y_data = np.linspace(0, 10, 10) + np.random.uniform(-1.5, 1.5, 10)

error = 0
m = tf.Variable(0.4)
b = tf.Variable(0.2)

for x,y in zip(x_data, y_data):
    y_hat = x * m + b
    error += (y - y_hat)**2

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
train = optimizer.minimize(error)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    
    epoches = 100
    for i in range(epoches):

        sess.run(train)
    
    slope, intercept = sess.run([m, b])
    print (slope)
    print (intercept)


# In[74]:


import matplotlib.pyplot as plt
x_test = np.linspace(-1, 11, 10)
y_pred = x_test * slope + intercept

plt.plot(x_test, y_pred, 'r')
plt.plot(x_data, y_data, '*')
plt.show()


# In[ ]:





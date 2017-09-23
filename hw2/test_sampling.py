import tensorflow as tf
import numpy as np

init_var = 1
ac_dim = 1
sy_mean = tf.constant(np.ones(ac_dim).reshape([1,ac_dim]), dtype=tf.float32)

# sy_logstd = tf.Variable(np.ones(ac_dim)*init_var, name='logstd' , dtype=tf.float32, trainable=True) # logstd should just be a trainable variable, not a network output.  #0.1
sy_logstd = tf.constant(np.ones(ac_dim), dtype=tf.float32)

sigma = tf.diag(tf.exp(sy_logstd))
sy_sampled_ac = sy_mean + tf.reshape(tf.matmul(sigma, tf.expand_dims(tf.random_normal(shape=[ac_dim]), 1)), [1, ac_dim])


sess = tf.Session()
sess.__enter__() # equivalent to `with sess:`
tf.global_variables_initializer().run() #pylint: disable=E1101
sess = tf.InteractiveSession()
samp = sess.run(sy_sampled_ac)

res = []
for i in range(10):
    res.append(sess.run(tf.random_normal(shape=[ac_dim])))

print("average {} std{}".format(np.mean(res), np.std(res)))

print(samp)
from scipy.stats import multivariate_normal
import numpy as np
import tensorflow as tf

x = np.array([1., 1.])


sigma = np.diag([0.5, 0.5])
mean = np.array([2.5, 2.5])

np_logprob = np.log(multivariate_normal.pdf(x, mean=mean, cov=sigma))

tf_sigma = tf.constant(sigma)
inv_sigma = tf.matrix_inverse(tf_sigma)
sy_mean = tf.reshape(tf.constant(mean),shape=[1,2])

sy_ac_na = tf.reshape(tf.constant(x), shape=[1,2])

sy_logprob_n = -0.5 * tf.reduce_sum(tf.multiply(sy_mean - sy_ac_na, tf.matmul(sy_mean - sy_ac_na, inv_sigma)), axis=1) - 0.5* tf.log(tf.matrix_determinant(2*np.pi*tf_sigma))


sess = tf.InteractiveSession()

logprob = sess.run(sy_logprob_n)

print(logprob -np_logprob)

import tensorflow as tf

# mnomial =tf.multinomial(tf.constant([[16., 10., 10., 10., 10.]]), 1)
mnomial =tf.multinomial(tf.log([[10., 10.]]), 1)


sess = tf.Session()
sess.__enter__()

for i in range(20):
    [res] = sess.run(mnomial)
    print(res)

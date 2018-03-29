import tensorflow as tf

input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)

output = tf.multiply(input1,input2)

with tf.Session() as sess:
    print(sess.run(output,feed_dict={input1:[7],input2:[2]}))
    # 现在官方建议Dataset,不建议feed_dict
    # mul已不能用，要用mutiply

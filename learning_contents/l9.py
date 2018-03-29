import tensorflow as tf
import numpy as np

# Save to file
# remember to define the same dtype shape when store
# W = tf.Variable([[1,2,3],[3,4,5]],dtype=tf.float32,name='weights')
# b = tf.Variable([[1,2,3]],dtype=tf.float32,name='biases')

# init = tf.global_variables_initializer()

# saver = tf.train.Saver()

# with tf.Session() as sess:
    # sess.run(init)
    # save_path = saver.save(sess,"my_net/save_net.ckpt") # ckpt 后缀,my_net要先定义与py同目录
    # print("Save to path:",save_path)
    
    
# restore variables
# redefine the same shape and same type for your variables
W = tf.Variable(np.arange(6).reshape((2,3)),dtype=tf.float32,name='weights')
b = tf.Variable(np.arange(3).reshape((1,3)),dtype=tf.float32,name='biases')


# not nees init step

saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess,"my_net/save_net.ckpt") # ckpt 后缀,my_net要先定义与py同目录
    print("weights:",sess.run(W))
    print("biases:",sess.run(b))
  

  
#   只能保存variables，之后再提取variables，NN的框架还是要再定义一遍
#   上面注释掉的是产生需要store的文件，下面没注释的是提取 

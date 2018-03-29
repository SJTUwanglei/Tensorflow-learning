import tensorflow as tf

matrix1 = tf.constant([[3,3]])
matrix2 = tf.constant([[2],[2]])
product = tf.matmul(matrix1,matrix2)    # matrix multiply np.dot(m1,m2)

# method 1
sess = tf.Session() # Session是一个object,要大写首字母，会话控制
result = sess.run(product)  # 每run一次才会执行一下结构
print(result)
sess.close()    # 更整洁更系统

# method 2
with tf.Session() as sess:
    result2 = sess.run(product)
    print(result2)

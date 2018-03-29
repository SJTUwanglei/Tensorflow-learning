import tensorflow as tf

state = tf.Variable(0,name='counter')
print(state.name)
one = tf.constant(1)

new_value = tf.add(state,one)
update = tf.assign(state,new_value)

# 初始化所有变量，sess激活，！！！must have if define variable
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)  # init必须要run一下
    for _ in range(3):
        sess.run(update)
        print (sess.run(state)) # 直接print state无效，必须要sess指针指上

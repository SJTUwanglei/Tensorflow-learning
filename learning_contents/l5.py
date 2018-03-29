import tensorflow as tf
import numpy as np
# import matplotlib.pyplot as plt

# 有loss图

# 下面n_layer报错是因为上面参数没有定义时加入n_layer
def add_layer(inputs, in_size, out_size, n_layer, activation_function=None):
    layer_name = 'layer%s' % n_layer
    with tf.name_scope('layer'):#默认是linear function
        with tf.name_scope(layer_name):
            Weights = tf.Variable(tf.random_normal([in_size, out_size], name='W'))
            tf.summary.histogram(layer_name+'/weights', Weights)
        # 用随机变量是因为生成初始变量比全0好很多，行和列数
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)    # 都建议初始不为0
            tf.summary.histogram(layer_name+'/biases', biases)
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.matmul(inputs, Weights) + biases
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)
        tf.summary.histogram(layer_name+'/weights', outputs)
        return outputs

# define placeholder for inputs to nework
# make up some real data
x_data = np.linspace(-1,1,300)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) - 0.5 + noise

# 这样操作的意义
with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32, [None, 1], name='x_input')   # dtype=float32
    ys = tf.placeholder(tf.float32, [None, 1], name='y_input')

# add hidden layer
l1 = add_layer(xs, 1, 10, n_layer=1, activation_function=tf.nn.relu)
# define output layer
prediction = add_layer(l1, 10, 1, n_layer=2, activation_function=None) # if里定义过None

# the error between prediction and real data 
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),
                reduction_indices=[1])) # 后面的reduce等等框架也可以加名字，train也可以
    # tf.scalar_summary('loss',loss)  # 标量，与histogram不同，loss在events里
    tf.summary.scalar('loss',loss)
    
with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

# Very import of sess.run(init)
init = tf.global_variables_initializer()
sess = tf.Session()
# merged = tf.merge_all_summaries()
merged = tf.summary.merge_all()
# writer = tf.train.SummaryWriter("l5_log",sess.graph) 2016-11-30更新移除
writer = tf.summary.FileWriter("l5_log",sess.graph)
sess.run(init)

for i in range(1000):
    sess.run(train_step, feed_dict={xs:x_data, ys:y_data})
    if i%50 ==0:
        result = sess.run(merged, feed_dict={xs:x_data, ys:y_data})
        writer.add_summary(result, i)

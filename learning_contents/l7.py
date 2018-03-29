# drop out

import tensorflow as tf
from sklearn.datasets import load_digits
# from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

# load data
digits = load_digits()
X = digits.data     # 加载0-9的数字的data
y = digits.target
y = LabelBinarizer().fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=3)

def add_layer(inputs, in_size, out_size, layer_name, activation_function=None):    #默认是linear function
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    # 用随机变量是因为生成初始变量比全0好很多，行和列数
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)    # 都建议初始不为0
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    Wx_plus_b = tf.nn.dropout(Wx_plus_b, keep_prob)
    
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    tf.summary.histogram(layer_name + '/outputs', outputs)
    # 这个只想输出cross_entropy，没有写上面这句histogram时一直报错，同时下面的
    # 在cross_entropy后面的scalar也需要加上
    return outputs

# define placeholder for inputs to network
keep_prob = tf.placeholder(tf.float32)  # keep probility,保持不被drop out的参数
xs = tf.placeholder(tf.float32, [None,64])  # 8x8
ys = tf.placeholder(tf.float32, [None,10])

# add output layer
l1 = add_layer(xs, 64, 50, 'l1', activation_function=tf.nn.tanh)   # 这里100过拟合
# 试过其他和不使用af，，在处理过程中信息被处理为NONE
prediction = add_layer(l1, 50, 10, 'l2', activation_function=tf.nn.softmax)


# the loss between prediction and real data
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction),
                            reduction_indices=[1]))
tf.summary.scalar('loss', cross_entropy)
train_step = tf.train.GradientDescentOptimizer(0.6).minimize(cross_entropy)

sess = tf.Session()
# merged = tf.merge_all_summaries()
merged = tf.summary.merge_all()
# summary writer goes in here
train_writer = tf.summary.FileWriter("l7_logs/train",sess.graph)
train_writer = tf.summary.FileWriter("l7_logs/test",sess.graph)

sess.run(tf.global_variables_initializer())

for i in range(500):
    sess.run(train_step, feed_dict={xs:X_train, ys:y_train, keep_prob: 0.5})
    if i%50 == 0:
        # record loss
        train_result = sess.run(merged, feed_dict={xs:X_train, ys:y_train, keep_prob:1})
        test_result = sess.run(merged, feed_dict={xs:X_test, ys:y_test, keep_prob:1})
        
        train_writer.add_summary(train_result, i)
        train_writer.add_summary(test_result, i)

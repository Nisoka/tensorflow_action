# coding: utf-8
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

'''
 莫烦大神 Tensorflow 例子 17 实例训练mnist and dropout
'''
# 1 添加层 内嵌很多<步骤>
'''
 l1 就是 add_layer 的输出值 
     inputs  输入数据
     in_size 输入层神经元 数目
     out_size 输出层神经元 数目
'''

def add_layer(inputs, in_size, out_size, n_layer, activation_function = None):
    # tf.Variable( tf.random_normal( [m, n] ) )
    # tf.random_noram( [m, n] ) 生成随机 mXn 矩阵
    # tf.zeros( [m, n] ) 生成 mXn 0矩阵
    layer_name = 'layer%d' % n_layer
    with tf.name_scope('layer'):
        Weights = tf.Variable(tf.random_normal([in_size, out_size]), name = 'b')
        biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, name = 'b')
        Wx_plus_b = tf.matmul(inputs, Weights) + biases
        Wx_plus_b = tf.nn.dropout(Wx_plus_b, keep_prob)
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)
        tf.histogram_summary(layer_name + '/outputs', outputs)

        return outputs

# 2 定义 <placeholder> (类型, [结构])
#     [None, 1] None 代表任意大小
keep_prob = tf.placeholder(tf.float32)
xs = tf.placeholder(tf.float32, [None, 784], name = 'x_input')
ys = tf.placeholder(tf.float32, [None, 10], name = 'y_input')

# 添加一个 隐藏层
# 添加一个 输出层
l1 = add_layer(xs, 784, 100, n_layer = 1, activation_function = tf.nn.tanh)
prediction = add_layer(l1, 100, 10, n_layer = 2, activation_function = tf.nn.softmax)

# reduce_mean 平均值
# reduce_sum 求和, reduction_indices = [1]??
# square 求平方
# 3 嵌入 <步骤>   prediction是一个runStruct, 会内嵌执行
cross_entroy = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction),
                                             reduction_indices = [1]))
tf.scalar_summary('loss', cross_entroy)   # tensorboard 的 Event 中
# 4 定义 <步骤> train.GradientDescentOptimizer(alpha) runStruct 进行学习
# 5 嵌入 <步骤> loss是一个runStruct, 会内嵌执行.
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entroy)


def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict = {xs: v_xs})
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys})
    return result

init = tf.initialize_all_variables()
merged = tf.merge_all_summaries()

sess = tf.Session()
train_writer = tf.train.SummaryWriter('logs/train', sess.graph)
test_writer = tf.train.SummaryWriter('logs/test', sess.graph)
sess.run(init)

for i in range(1200):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={xs:batch_xs, ys:batch_ys, keep_prob:0.5})
    if i%50 == 0:
        # record loss
        train_result = sess.run(merged, feed_dict={xs:batch_xs, ys:batch_ys, keep_prob:1})
        test_result = sess.run(merged, feed_dict={xs:mnist.test.images, ys:mnist.test.labels, keep_prob:1})
        train_writer.add_summary(train_result, i)
        test_writer.add_summary(test_result, i)

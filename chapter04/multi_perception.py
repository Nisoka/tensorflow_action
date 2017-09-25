from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

mnist = input_data.read_data_sets("../MNIST_data/", one_hot = True)
sess = tf.InteractiveSession()


in_units = 784
h1_units = 300
W1 = tf.Variable(tf.truncated_normal([in_units, h1_units], stddev=0.1))
b1 = tf.Variable(tf.zeros([h1_units]))
W2 = tf.Variable(tf.zeros([h1_units, 10]))
b2 = tf.Variable(tf.zeros([10]))

x = tf.placeholder(tf.float32, [None, in_units])
keep_prob = tf.placeholder(tf.float32)

# 激活函数 relu 梯度计算更好.
# hidden_1 计算子图
hidden_1 = tf.nn.relu(tf.matmul(x, W1) + b1)
# drop_out 舍弃掉hidden_1中的一些神经元, 这样每次进行拟合梯度下降, 都会得到不同的神经网络
# 最终得到一个取平均 得到一个最终的神经网络, 可以有效的防止过拟合。
hidden_1_drop = tf.nn.dropout(hidden_1, keep_prob)

y = tf.nn.softmax(tf.matmul(hidden_1_drop, W2) + b2)
y_ = tf.placeholder(tf.float32, [None, 10])

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y),
                                              reduction_indices = [1]))
train_step = tf.train.AdagradOptimizer(0.3).minimize(cross_entropy)
variable_init = tf.global_variables_initializer()

sess.run(variable_init)

for i in range(300):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    train_step.run({x: batch_xs, y_:batch_ys, keep_prob:0.75})

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(accuracy.eval({x:mnist.test.images, y_:mnist.test.labels, keep_prob:1.0}))

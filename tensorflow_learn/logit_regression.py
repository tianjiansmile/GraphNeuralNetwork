import tensorflow as tf
import matplotlib.pyplot as plt, matplotlib.image as mpimg
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data/',one_hot=True)

batch_size = 100
x = tf.placeholder(tf.float32, [None, 784], name='X')
y = tf.placeholder(tf.float32, [None, 10], name = 'Y')


# 模型参数 把784个像素看成单个特征，因为是多分类问题 [55000, 784]* w = [55000, 10]
W = tf.Variable(tf.zeros([784,10]), name = 'W')
b = tf.Variable(tf.zeros([10]), name= 'b')

# 预测函数
y_ = tf.matmul(x,W) + b
pred = tf.nn.softmax(y_)

# softmax交叉熵损失函数 自定义
loss = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred),1))

# y = np.array(y).astype(np.float64)
# loss = tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=y_)

# 优化器  AdagradOptimizer实际上属于自适应的梯度下降算法
opt = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)

    # 开始训练
    for epoch in range(50):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples/batch_size)
        for i in range(total_batch):
            batch_xs,batch_ys = mnist.train.next_batch(batch_size)
            sess.run(opt,feed_dict={x:batch_xs,y:batch_ys})

            avg_cost += sess.run(loss, feed_dict={x:batch_xs,y:batch_ys}) / total_batch

        # if (epoch + 1) % 5 == 0:
        print('avg cost',avg_cost)

    print('done')



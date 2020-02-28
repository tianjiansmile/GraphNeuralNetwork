import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 数据标准化，转换为正态分布
def normalize(X):
    mean = np.mean(X)
    std = np.std(X)
    X = (X - mean) / std
    return X

def append_bias_reshape(features, labels):
    m = features.shape[0]
    n = features.shape[1]
    x = np.reshape(np.c_[np.ones(m),features],[m,n+1])
    y = np.reshape(labels, [m,1])
    return x, y

# 读取数据
boston = tf.contrib.learn.datasets.load_dataset('boston')
X_train, Y_train = boston.data, boston.target
X_train = normalize(X_train)
m = len(X_train)
n = 13


# 占位符
X = tf.placeholder(tf.float32, name='X', shape=[m,n])
Y = tf.placeholder(tf.float32, name='Y')

# 声明参数
w = tf.Variable(tf.random_normal([n,1]), name='w')
b = tf.Variable(1.0, name='b')

# 预测函数
Y_hat = tf.matmul(X, w) + b

# reduce_mean 平均值 square平方和  reduce_sum 总和

# 损失函数
# loss = tf.reduce_mean(tf.square(Y - Y_hat, name='loss'))
# 目标函数
loss = tf.reduce_mean(tf.square(Y - Y_hat, name='loss')) + 0.6 * tf.nn.l2_loss(w)

# 梯度下降优化器
opt = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)

# 初始化操作符
init_op = tf.global_variables_initializer()
total = []

# 开始计算图
with tf.Session() as sess:
    sess.run(init_op)
    writer = tf.summary.FileWriter('graphs', sess.graph)

    for i in range(50):
        _, l = sess.run([opt, loss], feed_dict = {X: X_train, Y: Y_train})

        # 将本轮loss记录下来
        total.append(l)
        print('Epoch {0}: Loss {1}'.format(i, l))
        writer.close()
        # 拿到最终学得的参数
        w_value, b_value = sess.run([w, b])

    # 预测函数表达
    # Y_pred = tf.matmul(X_train, w_value) + b_value
    print('Done')

    # 损失值变化
    plt.plot(total)
    plt.show()











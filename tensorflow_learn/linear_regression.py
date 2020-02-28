import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 数据标准化，转换为正态分布
def normalize(X):
    mean = np.mean(X)
    std = np.std(X)
    X = (X - mean) / std
    return X

# 读取数据
boston = tf.contrib.learn.datasets.load_dataset('boston')
X_train, Y_train = boston.data[:,5],boston.target
n_samples = len(X_train)

# 占位符
X = tf.placeholder(tf.float32, name='X')
Y = tf.placeholder(tf.float32, name='Y')

b = tf.Variable(0.0)
w = tf.Variable(0.0)

# 预测函数
Y_hat = X * w + b
# 损失函数
loss = tf.square(Y - Y_hat, name='loss')

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
        total_loss = 0
        for x,y in zip(X_train, Y_train):
            _, l = sess.run([opt, loss],
            feed_dict = {X: x, Y: y})

            # print(_, l)
            total_loss += l
        # 将本轮平均的loss记录下来
        total.append(total_loss / n_samples)
        print('Epoch {0}: Loss {1}'.format(i, total_loss/n_samples))
        writer.close()
        # 拿到最终学得的参数
        b_value, w_value = sess.run([b,w])

    # 预测函数表达
    Y_pred = X_train * w_value + b_value
    print('Done')

    plt.plot(X_train, Y_train, 'bo', label='Real Data')
    plt.plot(X_train, Y_pred, 'r', label='Predict Data')
    plt.legend()
    plt.show()
    # 损失值变化
    plt.plot(total)
    plt.show()











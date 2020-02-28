import tensorflow as tf

# 定义常量
message = tf.constant('Welcom to the tensorflow world')

# with tf.Session() as sess:
#     print(sess.run(message).decode())

a = tf.constant([1.0,2.0,3.0],shape = [3], name='a')
b = tf.constant([1.0,2.0,3.0], shape = [3], name='b')
c = a +b
sess = tf.Session(config = tf.ConfigProto(log_device_placement =True))
print(sess.run(c))

# 计算图：是包含节点和边的网络。本节定义所有要使用的数据，也就是张量（tensor）对象（常量、变量和占位符），
# 同时定义要执行的所有计算，即运算操作对象（Operation Object，简称 OP）。

# 在此以两个向量相加为例给出计算图。假设有两个向量 v_1 和 v_2 将作为输入提供给 Add 操作
v1 = tf.constant([1,2,3,4])
v2 = tf.constant([2,1,5,4])
# v_add = tf.add(v1,v2)

# with tf.Session() as sess:
#     print(sess.run(v_add))

# 更简洁的方式
# print(tf.Session().run(tf.add(tf.constant([1,2,3]),tf.constant([4,5,6]))))

# Tensorflow 常量

# 零元素张量
# zero_t = tf.zeros([2,3],tf.int32)
# one_t = tf.ones([2,3],tf.int32)

# 使用以下语句创建一个具有一定均值（默认值=0.0）和标准差（默认值=1.0）、形状为 [M，N] 的正态分布随机数组
t_random = tf.random_normal([2,3],mean=2.0,stddev=4, seed=12)
# with tf.Session() as sess:
#     print(sess.run(t_random))

config = tf.ConfigProto(log_device_placement=True)

# TensorFlow 占位符
# 我们来讲解最重要的元素——占位符，它们用于将数据提供给计算图
# 使用 feed_dict 输入一个随机的 4×5 矩阵
x = tf.placeholder("float")
y = 2 * x
# data = tf.random_uniform([4,5],10)
# with tf.Session() as sess:
#     x_data = sess.run(data)
#     print(sess.run(y, feed_dict = {x:x_data}))
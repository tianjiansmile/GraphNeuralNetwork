import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D
from keras.datasets import imdb
# 每个例子包含一句电影评论和对应的标签，0或1。0代表负向评论，1代表正向评论。


max_features = 5000
# 同一单条文本长度400， 不足的用0补齐
maxlen = 400

# 训练测试数据，分别有25000条评论
# 这里将文字转换为数字id
print('Loading data...')
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

# 编号转换为原文
# print(decode_review(x_train[0]))

# print('第0条长度',len(x_train[0]), x_train[0], y_train[0])
# print('第5条长度',len(x_train[5]), x_train[5], y_train[5])

print(' 将每个评论补齐， 默认从前面补齐， Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
# print('第0条长度',len(x_train[0]), x_train[0], y_train[0])
# print('第5条长度',len(x_train[5]), x_train[5], y_train[5])


def weight_var(shape):

    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))

def bias_var(shape):
    return tf.Variable(tf.constant(0.1, shape=shape))

def conv_op(x,w):

    return tf.nn.conv2d(x, w, strides=[1,1,1,1], padding="SAME")

def max_pool_2x_op(x):

    return tf.nn.max_pool(x, strides=[1,2,2,1], ksize=[1,2,2,1])

def train(mnist):

    x = tf.placeholder(tf.float32, shape=[None, 784])
    y = tf.placeholder(tf.float32, shape=[None, 10])

    x_image = tf.reshape(x, [-1, 28, 28, 1])

    w_conv1 = weight_var([5,5,1,32])
    b_conv1 = bias_var([32])

    conv1 = conv_op(x_image,w_conv1) + b_conv1
    h_conv1 = tf.nn.relu(conv1)
    h_pool1 = max_pool_2x_op(h_conv1)

    w_fc1 = weight_var([7*7*32, 1024])
    b_fc1 = bias_var([1024])
    h_pool2 = tf.reshape(h_pool1,[-1, 7*7*32])

    h_fc1 = tf.nn.relu(tf.matmul(h_pool2,w_fc1)+ b_fc1)

    keep_prob = tf.placeholder(float)
    h_fc_dropout = tf.nn.dropout(h_fc1,keep_prob)

    w_fc2 = weight_var([1021,10])
    b_fc2 = bias_var([10])
    h_fc2 = tf.nn.softmax(tf.matmul(h_fc_dropout,w_fc2) + b_fc2)

    loss = tf.reduce_mean(-tf.reduce_sum(y*tf.log(h_fc2), reduction_indices=1))

    opt = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        for i in range(200):
            for i in





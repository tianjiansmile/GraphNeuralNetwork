import tensorflow as tf
import tflearn

from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D


from keras.datasets import imdb
# 每个例子包含一句电影评论和对应的标签，0或1。0代表负向评论，1代表正向评论。


def decode_review(text):
    # A dictionary mapping words to an integer index
    word_index = imdb.get_word_index()

    # The first indices are reserved
    word_index = {k: (v + 3) for k, v in word_index.items()}
    word_index["<PAD>"] = 0
    word_index["<START>"] = 1
    word_index["<UNK>"] = 2  # unknown
    word_index["<UNUSED>"] = 3

    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

    return ' '.join([reverse_word_index.get(i, '?') for i in text])

#

# set parameters:

max_features = 5000
# 同一单条文本长度400， 不足的用0补齐
maxlen = 400

batch_size = 32
embedding_dims = 50

filters = 250
kernel_size = 3
hidden_dims = 250
epochs = 2


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

print('Build model...')
model = Sequential()

# we start off with an efficient embedding layer which maps
# our vocab indices into embedding_dims dimensions
model.add(Embedding(max_features,
                    embedding_dims,
                    input_length=maxlen))
model.add(Dropout(0.2))

# we add a Convolution1D, which will learn filters
# word group filters of size filter_length:
# Conv1D  一维卷积，并不是说卷积核是一维的，而是说卷积操作只在纵列进行，得到的卷积层是一维的
# 典型的就是自然语言相关的神经网络，序列数据的操作
# Conv2D 二维卷积一般用于图像方面
model.add(Conv1D(filters,
                 kernel_size,
                 padding='valid',
                 activation='relu',
                 strides=1))
# we use max pooling:
model.add(GlobalMaxPooling1D())

# We add a vanilla hidden layer:
model.add(Dense(hidden_dims))
model.add(Dropout(0.2))
model.add(Activation('relu'))

# We project onto a single unit output layer, and squash it with a sigmoid:
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test))

# trainX = pad_sequences(trainX, maxlen=100, value=0.)
# testX = pad_sequences(testX, maxlen=100, value=0.)
#
# trainY = to_categorical(trainY)
# testY = to_categorical(testY)
#
#
# network = intput_data(shape=[None, 100], name='input')
# network = tflearn.embedding(network, input_dim=10000, output_dim=128)
#
# branch1 = conv_1d(network, 128, 3 ,padding='valid', activation='relu',
#                   regularizer='L2')
#
# branch2 = conv_1d(network, 128, 4 ,padding='valid', activation='relu',
#                   regularizer='L2')
#
# branch3 = conv_1d(network, 128, 5 ,padding='valid', activation='relu',
#                   regularizer='L2')
#
# network = merge([branch1, branch2, branch3], mode='concat', axis=1)
# network = tf.expand_dims(network, 2)
# network = global_max_pool(network)
# network = dropout(network, 0.5)
# network = fully_connected(network, 2, activation='softmax')
#
#

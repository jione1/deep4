#numpy로 데이터 읽어와서 분류

import tensorflow as tf
import numpy as np
import time
import random

import matplotlib  
# matplotlib.use('TkAgg')   
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from matplotlib import rc

rc('font', family='AppleGothic')
plt.rcParams['axes.unicode_minus'] = False



food = np.load('food64.npy')
# food = np.load('/Users/jisoo/Desktop/food64.npy')

np.random.shuffle(food)

row = food.shape[0]
train_num = int(row*0.75)
#  print(train_num) 7459개
# 총 12432
# print(food.shape[0])

# 64*64*3 = 12288
x_train = food[:train_num, :12288]
x_test = food[train_num:, :12288]
# print(x_train.shape)
y_train = food[:train_num, 12288:]
y_test = food[train_num:, 12288:]
# print(y_train.shape)


learning_rate = 0.01
batch_size = 500
training_epochs = 1000


global_step = tf.Variable(0, trainable=False, name='global_step')
keep_prob = tf.placeholder(tf.float32)


def batch_norm(input_layer, scope):
    BN_EPSILON = 1e-5
    dimension = int(input_layer.shape[3])
    with tf.variable_scope(scope):
        mean, variance = tf.nn.moments(input_layer, axes=[0, 1, 2])
        beta = tf.get_variable('beta', dimension, tf.float32,
                initializer=tf.constant_initializer(0.0, tf.float32))
        gamma = tf.get_variable('gamma', dimension, tf.float32,
                initializer=tf.constant_initializer(1.0, tf.float32))
        bn_layer = tf.nn.batch_normalization(input_layer, mean, variance, beta, gamma, BN_EPSILON)
                 
    return bn_layer


# X = tf.placeholder(tf.float32, shape=[None, 64, 64, 3])
X = tf.placeholder(tf.float32, [None, 12288])
x_img = tf.reshape(X, [-1, 64, 64, 3]) 
Y = tf.placeholder(tf.float32, [None, 15])


# 한번에 볼 사이즈, 입력값, 출력값
W1 = tf.Variable(tf.random_normal(shape=[5,5,3,32], stddev=5e-2))
L1 = tf.nn.conv2d(x_img, W1, strides = [1,1,1,1], padding='SAME')
bn_layer_1 = batch_norm(L1, 'batch1')
L1 = tf.nn.relu(bn_layer_1)
L1 = tf.nn.max_pool(L1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
L1 = tf.nn.dropout(L1, keep_prob=keep_prob)
# print(L1.get_shape())


W2 = tf.Variable(tf.random_normal(shape=[3,3,32,64], stddev=5e-2))
L2 = tf.nn.conv2d(L1, W2, strides = [1,1,1,1], padding='SAME')
bn_layer_2 = batch_norm(L2, 'batch2')
L2 = tf.nn.relu(bn_layer_2)
L2 = tf.nn.max_pool(L2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
L2 = tf.nn.dropout(L2, keep_prob=keep_prob)
# print(L2.get_shape())

W = tf.Variable(tf.random_normal(shape=[3,3,64,64], stddev=5e-2))
L = tf.nn.conv2d(L2, W, strides = [1,1,1,1], padding='SAME')
bn_layer_ = batch_norm(L, 'batch')
L = tf.nn.relu(bn_layer_)
L = tf.nn.max_pool(L, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
L = tf.nn.dropout(L, keep_prob=keep_prob)


W3 = tf.Variable(tf.random_normal(shape=[3,3,64,128], stddev=5e-2))
L3 = tf.nn.conv2d(L, W3, strides = [1,1,1,1], padding='SAME')
bn_layer_3 = batch_norm(L3, 'batch3')
L3 = tf.nn.relu(bn_layer_3)
L3 = tf.nn.max_pool(L3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
L3 = tf.nn.dropout(L3, keep_prob=keep_prob)

L3_flat = tf.reshape(L3, [-1,128*4*4])

# 625는 내맘대로 정한 아웃풋
W4 = tf.get_variable("W4", shape=[128*4*4, 425],
                    initializer=tf.contrib.layers.xavier_initializer())
b4 = tf.Variable(tf.random_normal([425]))
L4 = tf.nn.relu(tf.matmul(L3_flat, W4)+b4)
L4 = tf.nn.dropout(L4, keep_prob=keep_prob)


W5 = tf.get_variable("W5", shape=[425, 15],
                     initializer=tf.contrib.layers.xavier_initializer())
b5 = tf.Variable(tf.random_normal([15]))
logits = tf.matmul(L4, W5) + b5
y_pred = tf.nn.softmax(logits)
# print(y_pred.get_shape())

def l2loss(var_list):
    regul = tf.nn.l2_loss(var_list[0])
    for v in var_list[1:]:
        regul += tf.nn.l2_loss(v)
    return regul

print('checkpoint')
t_vars = tf.trainable_variables()
print(t_vars)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
cost += 1e-4 * l2loss(t_vars)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost,global_step=global_step)

correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

count = int(train_num / batch_size)
if train_num % batch_size != 0:
    count += 1

with tf.Session() as sess:


    # checkpoint
    
    saver = tf.train.Saver(tf.global_variables())
    ckpt = tf.train.get_checkpoint_state('~/ckpt_food')
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        sess.run(tf.global_variables_initializer())


    print('Learning started. It takes sometime.')

    learning_start = time.time()
    learning_start_time = time.strftime("%X", time.localtime())

    for i in range(training_epochs):

        for b in range(count):
            b_count = b*batch_size

            x = x_train[b_count:b_count+batch_size,:]
            y = y_train[b_count:b_count+batch_size,:]
            # print("b    ",b)

            if b == count-1 :
                x = x_train[b_count:,:]
                y = y_train[b_count:,:]

            feed_dict = {X : x, Y : y, keep_prob: 0.5}
            a, c, _ = sess.run([accuracy, cost, optimizer], feed_dict=feed_dict)

        print('Epoch:', '%04d' % (i), 'cost =', '{:.9f}'.format(c), ' accuracy =', '{:.9f}'.format(a))
        
        # saver.save(sess, '~/ckpt_food/cnn_food.ckpt', global_step=global_step)

    print('Learning Finished!')

    learning_end = time.time()
    learning_end_time = time.strftime("%X", time.localtime())
    min = (learning_end - learning_start)/60
    print('%s ~ %s, 소요시간: %s분' %(learning_start_time, learning_end_time, '{:.1f}'.format(min)))

    acc = sess.run(accuracy, feed_dict={X: x_test, Y: y_test, keep_prob: 1.0})
    print('Accuracy:','{:.2f}'.format(acc*100), '%')


    #사진으로 보여주기
    #len => 리스트 사이즈 range=> 지정한 숫자만큼 배열을 만들어줌
    labels = sess.run(logits, feed_dict={X: x_test, keep_prob: 1})
    random_idxs = random.sample(range(len(food)), 10)

    fig = plt.figure()
    for i, r in enumerate(random_idxs):
        subplot = fig.add_subplot(2, 10, i+1)
        subplot.set_xticks([])
        subplot.set_yticks([])
        food_list = ['감자','계란','당근','무','소갈비','양상추','양파','치즈',
        '토마토','파','패티','표고버섯','피클','햄','햄버거빵']
        print(food_list[np.argmax(labels[i])])
        subplot.set_title(food_list[np.argmax(labels[i])])

        subplot.imshow(np.float64((x_test[i].reshape(64, 64,3))/255.0))

    plt.show()
  
#Deep CNN
import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt

def batch_norm(input_layer, scope):
	BN_EPSILON = 1e-5
	dimension = int(input_layer.shape[3])
	with tf.variable_scope(scope):
		mean, variance = tf.nn.moments(input_layer, axes=[0, 1, 2])
		beta = tf.get_variable('beta', dimension, tf.float32, initializer = tf.constant_initializer(0.0, tf.float32))
		gamma = tf.get_variable('gamma', dimension, tf.float32, initializer = tf.constant_initializer(1.0, tf.float32))
		bn_layer = tf.nn.batch_normalization(input_layer, mean, variance, beta, gamma, BN_EPSILON)

	return bn_layer

def l2loss(var_list):
    regul = tf.nn.l2_loss(var_list[0])
    for v in var_list[1:]:
        regul += tf.nn.l2_loss(v)
    return regul

training_data = np.load('training_data.npy')
test_data = np.load('test_data.npy')

tf.set_random_seed(777)

learning_rate = 0.0001
training_epochs = 300
batch_size = 100
beta = 0.0005

keep_prob = tf.placeholder(tf.float32)

X = tf.placeholder(tf.float32, [None, 12288])
X_img = tf.reshape(X, [-1, 64, 64, 3])
Y = tf.placeholder(tf.float32, [None, 15])

W1 = tf.Variable(tf.random_normal([3, 3, 3, 16], stddev=0.01))
L1 = tf.nn.conv2d(X_img, W1, strides=[1, 1, 1, 1], padding='SAME')
print("Conv2d: ", L1)

L1 = batch_norm(L1, 'L1')

L1 = tf.nn.relu(L1)
print("Relu: ", L1)

L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
print("MaxPool: ", L1)

L1 = tf.nn.dropout(L1, keep_prob=keep_prob)
print("Dropout: ", L1)

W2 = tf.Variable(tf.random_normal([3, 3, 16, 32], stddev=0.01))
L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
print("Conv2d: ", L2)

L2 = batch_norm(L2, 'L2')

L2 = tf.nn.relu(L2)
print("Relu; ", L2)

L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
print("MaxPool: ", L2)

L2 = tf.nn.dropout(L2, keep_prob=keep_prob)
print("Dropout: ", L2)

W3 = tf.Variable(tf.random_normal([3, 3, 32, 128], stddev=0.01))
L3 = tf.nn.conv2d(L2, W3, strides=[1, 1, 1, 1], padding='SAME')
print("Conv2d: ", L3)

L3 = batch_norm(L3, 'L3')

L3 = tf.nn.relu(L3)
print("Relu: ", L3)

L3 = tf.nn.max_pool(L3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
print("MaxPool:", L3)

L3 = tf.nn.dropout(L3, keep_prob=keep_prob)
print("Dropout: ", L3)

W4 = tf.Variable(tf.random_normal([3, 3, 128, 256], stddev=0.01))
L4 = tf.nn.conv2d(L3, W4, strides=[1, 1, 1, 1], padding='SAME')

L4 = batch_norm(L4, 'L4')

L4 = tf.nn.relu(L4)
L4 = tf.nn.max_pool(L4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
L4 = tf.nn.dropout(L4, keep_prob=keep_prob)
L4_flat = tf.reshape(L4, [-1, 128 * 4 * 4 * 2])
print("Reshape: ", L4_flat)

W5 = tf.get_variable("W5", shape=[128 * 4 * 4 * 2, 128], initializer=tf.contrib.layers.xavier_initializer())
b5 = tf.Variable(tf.random_normal([128]))
L5 = tf.nn.relu(tf.matmul(L4_flat, W5) + b5)

L5 = tf.nn.dropout(L5, keep_prob=keep_prob)
print("L5", L5)


W6 = tf.get_variable("W6", shape=[128, 15], initializer=tf.contrib.layers.xavier_initializer())
b6 = tf.Variable(tf.random_normal([15]))
hypothesis = tf.matmul(L5, W6) + b6
print("hypothesis", hypothesis)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=hypothesis, labels=Y))

t_vars = tf.trainable_variables()
cost += beta * l2loss(t_vars)

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)


correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

print('Learning started. It takes sometime.')
for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = int(len(training_data) / batch_size)

    for i in range(total_batch):
        batch_xs = training_data[i * batch_size:(i + 1) * batch_size, :-15]
        batch_ys = training_data[i * batch_size:(i + 1) * batch_size, -15:]

        c, _ = sess.run([cost, optimizer], feed_dict={X: batch_xs, Y: batch_ys, keep_prob: 0.7})
        avg_cost += c / total_batch

    print('Epoch:', '%04d' % (epoch + 1), 'cost = ', '{:.9f}'.format(avg_cost))
    acc = sess.run(accuracy, feed_dict={X: training_data[:, :-15], Y: training_data[:, -15:], keep_prob: 1})
    print('Accuracy: ', '{:.2f}'.format(acc*100))

print('Learning Finished!')

acc = sess.run(accuracy, feed_dict={X: test_data[:, :-15], Y: test_data[:, -15:], keep_prob: 1})
print('Accuracy: ', '{:.2f}'.format(acc*100))


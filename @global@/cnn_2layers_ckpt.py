import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import random

training_data = np.load('model_data/training_data.npy')
test_data = np.load('model_data/test_data.npy')

# hyperparameters
learning_rate = 0.001
training_epochs = 5
batch_size = 100

global_step = tf.Variable(0, trainable=False, name='global_step')
keep_prob = tf.placeholder(tf.float32)

X = tf.placeholder(tf.float32, [None, 1800])
X_img = tf.reshape(X, [-1, 45, 40, 1])
Y = tf.placeholder(tf.float32, [None, 4])

# conv layer 1
W1 = tf.Variable(tf.random_normal([3, 3, 1, 16], stddev=0.01))
L1 = tf.nn.conv2d(X_img, W1, strides=[1, 1, 1, 1], padding='SAME')
L1 = tf.nn.relu(L1)
L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1],
                    strides=[1, 2, 2, 1], padding='SAME')
L1 = tf.nn.dropout(L1, keep_prob=keep_prob)

# conv layer 2
W2 = tf.Variable(tf.random_normal([3, 3, 16, 32], stddev=0.01))
L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
L2 = tf.nn.relu(L2)
L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1],
                    strides=[1, 2, 2, 1], padding='SAME')
L2 = tf.nn.dropout(L2, keep_prob=keep_prob)
L2_flat = tf.reshape(L2, [-1, 32 * 12 * 10])

# fc layer 1
W3 = tf.get_variable("W3", shape=[32 * 12 * 10, 125],
                     initializer=tf.contrib.layers.xavier_initializer())
b3 = tf.Variable(tf.random_normal([125]))
L3 = tf.nn.relu(tf.matmul(L2_flat, W3) + b3)
L3 = tf.nn.dropout(L3, keep_prob=keep_prob)

# fc layer 2
W4 = tf.get_variable("W4", shape=[125, 4],
                     initializer=tf.contrib.layers.xavier_initializer())
b4 = tf.Variable(tf.random_normal([4]))
logits = tf.matmul(L3, W4) + b4

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost, global_step=global_step)

sess = tf.Session()
saver = tf.train.Saver(tf.global_variables())

# checkpoint
ckpt = tf.train.get_checkpoint_state('./ckpt_4labels')
if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
	saver.restore(sess, ckpt.model_checkpoint_path)
else:
	sess.run(tf.global_variables_initializer())

# training
print('학습 시작!!!')
learning_start = time.time()
learning_start_time = time.strftime("%X", time.localtime())

for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = int(len(training_data) / batch_size)

    for i in range(total_batch):
        batch_xs = training_data[i*batch_size:(i+1)*batch_size, :-4]
        batch_ys = training_data[i*batch_size:(i+1)*batch_size, -4:]
        c, _ = sess.run([cost, optimizer], feed_dict={X: batch_xs, Y: batch_ys, keep_prob: 0.7})
        avg_cost += c / total_batch

    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))
    saver.save(sess, './ckpt_4labels/cnn_4labels.ckpt', global_step=global_step)

print('학습 끝!!!!!')
learning_end = time.time()
learning_end_time = time.strftime("%X", time.localtime())
print('%s ~ %s, 소요시간: %s초' %(learning_start_time, learning_end_time, learning_end - learning_start))

# testing
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print('Accuracy:', sess.run(accuracy, feed_dict={
      X: test_data[:, :-4], Y: test_data[:, -4:], keep_prob: 1}))

# matplotlib
labels = sess.run(logits, feed_dict={X: test_data[:, :-4], keep_prob: 1})
random_idxs = random.sample(range(len(test_data)), 20)

fig = plt.figure()
for i, r in enumerate(random_idxs):
    subplot = fig.add_subplot(2, 10, i+1)
    subplot.set_xticks([])
    subplot.set_yticks([])
    if np.argmax(labels[i]) == 0:
        subplot.set_title('고딕')
    elif np.argmax(labels[i]) == 1:
        subplot.set_title('명조')
    elif np.argmax(labels[i]) == 2:
        subplot.set_title('손글씨')
    elif np.argmax(labels[i]) == 3:
        subplot.set_title('장식')
    subplot.imshow(test_data[:, :-4][i].reshape(45, 40), cmap='gray')
plt.show()
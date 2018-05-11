import tensorflow as tf
import numpy as np

from tensorflow.python.keras._impl.keras.datasets.cifar10 import load_data

tf.set_random_seed(777)
learning_rate = 0.001
training_epochs = 1000
batch_size = 256


def next_batch(num, data, labels):
    idx = np.arange(0, len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[i] for i in idx]
    labels_shuffle = [labels[i] for i in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)


X = tf.placeholder(tf.float32, [None, 32, 32, 3])
X_img = X
Y = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)

W1 = tf.Variable(tf.random_normal([5, 5, 3, 64], stddev=0.01))
L1 = tf.nn.conv2d(X_img, W1, strides=[1, 1, 1, 1], padding='SAME')
print("L1 Cnv2d: ", L1)
L1 = tf.nn.relu(L1)
print("L1 Relu: ", L1)
L1 = tf.nn.max_pool(L1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
print("L1 MaxPool: ", L1)
L1 = tf.nn.dropout(L1, keep_prob=keep_prob)
print("L1 Dropout: ", L1)

W2 = tf.Variable(tf.random_normal([5, 5, 64, 64], stddev=0.01))
L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
print("L2 Conv2d: ", L2)
L2 = tf.nn.relu(L2)
print("L2 Relu: ", L2)
L2 = tf.nn.max_pool(L2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
print("L2 MaxPool: ", L2)
L2 = tf.nn.dropout(L2, keep_prob=keep_prob)
print("L2 Dropout: ", L2)

W3 = tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=0.01))
L3 = tf.nn.conv2d(L2, W3, strides=[1, 1, 1, 1], padding='SAME')
print("L3 Conv2d: ", L3)
L3 = tf.nn.relu(L3)
print("L3 Relu: ", L3)
L3 = tf.nn.max_pool(L3, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
print("L3 MaxPool: ", L3)
L3 = tf.nn.dropout(L3, keep_prob=keep_prob)
print("L3 Dropout: ", L3)

W4 = tf.Variable(tf.random_normal([3, 3, 128, 128], stddev=0.01))
L4 = tf.nn.conv2d(L3, W4, strides=[1, 1, 1, 1], padding='SAME')
print("L4 Conv2d: ", L4)
L4 = tf.nn.relu(L4)
print("L4 Relu: ", L4)
L4 = tf.nn.max_pool(L4, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
print("L4 MaxPool: ", L4)
L4 = tf.nn.dropout(L4, keep_prob=keep_prob)
print("L4 Dropout: ", L4)

L4_flat = tf.reshape(L4, [-1, 2 * 2 * 128])
print("L4 Reshape: ", L4)

W5 = tf.get_variable("W5", shape=[2 * 2 * 128, 384], initializer=tf.contrib.layers.xavier_initializer())
b5 = tf.Variable(tf.random_normal([384]))
L5 = tf.nn.relu(tf.matmul(L4_flat, W5) + b5)
print("L5 Relu: ", L5)
L5 = tf.nn.dropout(L5, keep_prob=keep_prob)
print("L5 Dropout: ", L5)

W6 = tf.get_variable("W6", shape=[384, 10], initializer=tf.contrib.layers.xavier_initializer())
b6 = tf.Variable(tf.random_normal([10]))
logits = tf.matmul(L5, W6) + b6
y_hat = tf.nn.softmax(logits)

(x_train, y_train), (x_test, y_test) = load_data()
y_train_one_hot = tf.squeeze(tf.one_hot(y_train, 10), axis=1)
y_test_one_hot = tf.squeeze(tf.one_hot(y_test, 10), axis=1)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

correct_prediction = tf.equal(tf.argmax(y_hat, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

print('Started')

for epoch in range(training_epochs):
    n_batches = 5
    for i in range(1, n_batches + 1):
        train_batch = next_batch(128, x_train, y_train_one_hot.eval())
        feed_dict = {X: train_batch[0], Y: train_batch[1], keep_prob: 0.7}
        a, c, _ = sess.run([accuracy, cost, optimizer], feed_dict=feed_dict)
    print('Epoch: ', '%04d' % (epoch + 1), 'Accuracy =', '{:.9f}'.format(a), 'cost =', '{:.9f}'.format(c))

print('Learning Finished!')

test_batch = next_batch(10000, x_test, y_test_one_hot.eval())
print('Accuracy:', sess.run(accuracy, feed_dict={X: test_batch[0], Y: test_batch[1], keep_prob: 1}))
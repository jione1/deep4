import tensorflow as tf
import numpy as np
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# CIFAR-10 데이터를 다운로드 받기 위한 helpder 모듈인 load_data 모듈을 임포트합니다.
from tensorflow.python.keras._impl.keras.datasets.cifar10 import load_data
(x_train, y_train), (x_test, y_test) = load_data()
y_train_one_hot = tf.squeeze(tf.one_hot(y_train, 10),axis=1)
y_test_one_hot = tf.squeeze(tf.one_hot(y_test, 10),axis=1)

def next_batch(num, data, labels):
    '''
    Return a total of `num` random samples and labels. 
    '''
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[ i] for i in idx]
    labels_shuffle = [labels[ i] for i in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)

learning_rate = 0.001
training_epochs = 15
batch_size = 100

keep_prob = tf.placeholder(tf.float32)

X = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
Y = tf.placeholder(tf.float32, [None, 10])
x_img = X

W1 = tf.Variable(tf.random_normal(shape=[3, 3, 3, 64], stddev=5e-2))
L1 = tf.nn.relu(tf.nn.conv2d(x_img, W1, strides = [1,1,1,1], padding='SAME'))
L1 = tf.nn.max_pool(L1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
L1 = tf.nn.dropout(L1, keep_prob=keep_prob)
print(L1.get_shape())

W2 = tf.Variable(tf.random_normal(shape=[3,3,64,64], stddev=5e-2))
L2 = tf.nn.relu(tf.nn.conv2d(L1, W2, strides = [1,1,1,1], padding='SAME'))
L2 = tf.nn.max_pool(L2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
L2 = tf.nn.dropout(L2, keep_prob=keep_prob)
print(L2.get_shape())


W3 = tf.Variable(tf.random_normal(shape=[3,3,64,128], stddev=5e-2))
L3 = tf.nn.relu(tf.nn.conv2d(L2, W3, strides = [1,1,1,1], padding='SAME'))
L3 = tf.nn.max_pool(L3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
L3 = tf.nn.dropout(L3, keep_prob=keep_prob)
print(L3.get_shape())


L3_flat = tf.reshape(L3, [-1,128*4*4])

W4 = tf.get_variable("W4", shape=[128*4*4, 625], 
                    initializer=tf.contrib.layers.xavier_initializer())
b4 = tf.Variable(tf.random_normal([625]))
L4 = tf.nn.relu(tf.matmul(L3_flat, W4)+b4)
L4 = tf.nn.dropout(L4, keep_prob=keep_prob)

W = tf.get_variable("W5", shape=[625, 10],
                     initializer=tf.contrib.layers.xavier_initializer())
b5 = tf.Variable(tf.random_normal([10]))
logits = tf.matmul(L4, W) + b5
y_pred = tf.nn.softmax(logits)
print(y_pred.get_shape())

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    print('Learning started. It takes sometime.')
    for i in range(10000):
            batch = next_batch(128, x_train, y_train_one_hot.eval())
            feed_dict = {X: batch[0], Y: batch[1], keep_prob: 0.7}
            a, c, _ = sess.run([accuracy, cost, optimizer], feed_dict=feed_dict)
            
            if i % 100 == 0:
                print('Epoch:', '%04d' % (i), 'cost =', '{:.9f}'.format(c), ' accauracy =', '{:.9f}'.format(a))

    print('Learning Finished!')

    # Test model and check accuracy

    # if you have a OOM error, please refer to lab-11-X-mnist_deep_cnn_low_memory.py


    test_batch = next_batch(10000, x_test, y_test_one_hot.eval())
    print('Accuracy:', sess.run(accuracy, feed_dict={
        X: test_batch[0], Y: test_batch[1], keep_prob: 1.0}))


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets(".mnist/data/", one_hot = True)

total_epoch = 100
batch_size = 200
learning_rate = 0.01
n_hidden = 256
n_input = 28*28
n_noise = 128
#노이즈 값을 랜덤이어도 됨

X = tf.placeholder(tf.float32, [None, n_input])
Z = tf.placeholder(tf.float32, [None, n_noise])

#생성자 신경망
G_W1 = tf.Variable(tf.random_normal([n_noise, n_hidden], stddev = 0.01))
G_b1 = tf.Variable(tf.zeros([n_hidden]))
G_W2 = tf.Variable(tf.random_normal([n_hidden, n_input], stddev = 0.01))
G_b2 = tf.Variable(tf.zeros([n_input]))

#구분자
D_W1 = tf.Variable(tf.random_normal([n_input, n_hidden], stddev = 0.01))
D_b1 = tf.Variable(tf.zeros([n_hidden]))
D_W2 = tf.Variable(tf.random_normal([n_hidden, 1], stddev = 0.01))
D_b2 = tf.Variable(tf.zeros([1]))

def generator(noize_z):
    hidden = tf.nn.relu(tf.matmul(noize_z, G_W1) + G_b1)
    output = tf.nn.sigmoid(tf.matmul(hidden, G_W2) + G_b2)
    return output

def get_noise(batch_size, n_noise):
    return np.random.normal(size = (batch_size, n_noise))


def discriminator(inputs):
    hidden = tf.nn.relu(tf.matmul(inputs, D_W1) + D_b1)
    output = tf.nn.sigmoid(tf.matmul(hidden, D_W2) + D_b2)
    return output


G = generator(Z)
D_gene = discriminator(G)
D_real = discriminator(X)

loss_D = tf.reduce_mean(tf.log(D_real) + tf.log(1 - D_gene))
loss_G = tf.reduce_mean(tf.log(D_gene))

D_var_list = [D_W1, D_b1, D_b2]
G_var_list = [G_W1, G_b1, G_b2]

train_D = tf.train.AdamOptimizer(learning_rate).minimize(loss_D, var_list=D_var_list)
train_G = tf.train.AdamOptimizer(learning_rate).minimize(loss_G, var_list=G_var_list)


sess = tf.Session()
sess.run(tf.global_variables_initializer())

total_batch = int(mnist.train.num_examples / batch_size)
loss_var_D , loss_var_G = 0,0

for epoch in range(total_epoch):
    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        noise = get_noise(batch_size, n_noise)

        _, loss_var_D = sess.run([train_D, loss_D],feed_dict = {X : batch_xs, Z : noise})
        _, loss_var_G = sess.run([train_G, loss_G],feed_dict = {Z : noise})

    print('Epoch:', '%04d' % epoch, 'D loss: {:.4}'.format(loss_var_D),'G loss: {:.4}'.format(loss_var_G))

    if epoch == 0 or (epoch + 1) % 10 ==0:
        sample_size = 10
        noise = get_noise(sample_size, n_noise)
        samples = sess.run(G, feed_dict = {Z : noise})

        fig,ax = plt.subplots(1, sample_size, figsize = (sample_size, 1))

        for i in range(sample_size):
            ax[i].set_axis_off()
            ax[i].imshow(np.reshape(samples[i], (28,28)))

        plt.savefig('samples/{}.png'.format(str(epoch).zfill(3)), bbox_inches='tight')
        plt.close(fig)

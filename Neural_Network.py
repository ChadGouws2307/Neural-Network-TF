# File Name

# @ Author: Chad Gouws
# Date: 28/03/2019

import tensorflow as tf
import numpy as np

# CONSTRUCTION

n_inputs = 28*28
n_hidden1 = 300
n_hidden2 = 200
n_hidden3 = 150
n_hidden4 = 100
n_hidden5 = 70
n_outputs = 10

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name='X')
y = tf.placeholder(tf.int64, shape=(None), name='y')


def neuron_layer(X, n_neurons, name, activation=None):
    with tf.name_scope(name):
        n_inputs = int(X.get_shape()[1])
        stddev = 2/np.sqrt(n_inputs)
        init = tf.truncated_normal((n_inputs, n_neurons), stddev=stddev)
        W = tf.Variable(init, name='kernel')
        b = tf.Variable(tf.zeros([n_neurons]), name='bias')
        Z = tf.matmul(X, W) + b

        if activation is not None:
            return activation(Z)
        else:
            return Z


with tf.name_scope('dnn'):
    hidden1 = neuron_layer(X, n_hidden1, name='hidden1', activation=tf.nn.elu)
    hidden2 = neuron_layer(hidden1, n_hidden2, name='hidden2', activation=tf.nn.elu)
    hidden3 = neuron_layer(hidden2, n_hidden3, name='hidden3', activation=tf.nn.elu)
    hidden4 = neuron_layer(hidden3, n_hidden4, name='hidden4', activation=tf.nn.elu)
    hidden5 = neuron_layer(hidden4, n_hidden5, name='hidden5', activation=tf.nn.elu)
    logits = neuron_layer(hidden5, n_outputs, name='outputs')

with tf.name_scope('loss'):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(xentropy, name='loss')

learning_rate = 0.01

with tf.name_scope('train'):
    optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
    training_op = optimizer.minimize(loss)

with tf.name_scope('eval'):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()
saver = tf.train.Saver()

# EXECUTION

import timeit
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('/tmp/data/')

n_epochs = 40
batch_size = 50

tic = timeit.default_timer()

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for iteration in range(mnist.train.num_examples // batch_size):
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})

        acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
        acc_test = accuracy.eval(feed_dict={X: mnist.test.images, y: mnist.test.labels})

        print(epoch, 'Train accuracy:', acc_train, 'Test accuracy:', acc_test)

    save_path = saver.save(sess, './my_model_final.ckpt')

toc = timeit.default_timer()

time = toc - tic
print('Process time:', time)



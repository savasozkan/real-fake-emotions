from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np

from compact_bilinear_pooling import compact_bilinear_pooling_layer

datau  = np.load('anger/data.npy') # upper face data
label  = np.load('anger/label.npy')      # labels 

datauval  = np.load('anger/dataval.npy') # upper face data
labelval  = np.load('anger/labelval.npy')      # labels 

# Parameters
learning_rate  = 0.001 
training_iters = 200000 
batch_size     = 64
display_step   = 10

# Network Parameters
n_input   = 256*6*6  # Dimension of emotion vector
n_steps   = 15         # Timesteps of LSTM
n_hidden  = 64      # LSTM hidden dimension
n_classes = 2         # Number of classes (real(1) or fake(0))
n_channel = 256

xu = tf.placeholder(tf.float32, [None, n_steps, n_input]) # upper face feature
y  = tf.placeholder("float", [None, n_classes])        # labels
kprobi = tf.placeholder("float", None)
kprobo = tf.placeholder("float", None)

# Define weights
W = {'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))}
b = {'out': tf.Variable(tf.random_normal([n_classes]))}

def cbp(x1, x2, in_size, dim):
      v  = compact_bilinear_pooling_layer(x1, x2, in_size, dim, sum_pool=True) 
      v.set_shape([batch_size, dim])
      
      return v
    
def RNNu(xu):
    xu_ = tf.unstack(xu, n_steps, 1)

    xux = []   
    for t in xrange(0, n_steps, 3):
        xmp = []

        xmp.append( tf.reshape(xu_[t+0], [-1, 36, n_channel]) )
        xmp.append( tf.reshape(xu_[t+1], [-1, 36, n_channel]) )
        xmp.append( tf.reshape(xu_[t+2], [-1, 36, n_channel]) )
	
	xmt = tf.transpose(xmp, perm=[1, 0, 2, 3])
        v = cbp(xmt, xmt, n_channel, n_hidden)
	
	xux.append( v )
 
    with tf.name_scope("fwu"),tf.variable_scope('fwu'): 
          lstm_u_cell  = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0, activation=tf.nn.relu)
          lstm_u_drop  = rnn.DropoutWrapper(lstm_u_cell, input_keep_prob=kprobi, output_keep_prob=kprobo)
          outputs_u, states_u = rnn.static_rnn(lstm_u_drop, xux, dtype=tf.float32)
	
    return tf.nn.l2_normalize(outputs_u[-1], dim=1)

def DNN(x, W, b):
    out = tf.matmul(x, W) + b
    return out

def prediction_loss(pred, y):
     return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))

pred  = RNNu(xu)
predp = DNN(pred, W['out'], b['out'])

cost = tf.multiply(1.0, prediction_loss(predp, y))

ibatch = tf.Variable(0)
ilr = tf.train.exponential_decay(learning_rate, ibatch*batch_size, 40000, 0.1, staircase=True) 

optimizer_cont = tf.train.AdamOptimizer(learning_rate=ilr, beta1=0.7).minimize(cost, global_step=ibatch)

correct_pred = tf.equal(tf.argmax(predp,1), tf.argmax(y,1))
accuracy     = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()

def OneHot(X, n=None, negative_class=0.):
    X = np.asarray(X).flatten()
    if n is None:
        n = np.max(X) + 1
    Xoh = np.ones((len(X), n)) * negative_class
    Xoh[np.arange(len(X)), X] = 1.
    return Xoh

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    step = 1

    while step * batch_size < training_iters:

        indices = np.random.choice(datau.shape[0], batch_size)
        batch_xu = datau[indices]
        batch_y  = label[indices]

        batch_xu           = batch_xu.reshape((batch_size, n_steps, n_input))
        batch_hoty         = OneHot(batch_y, 2)

        indices = np.random.choice(datauval.shape[0], batch_size)
        batch_xuval = datauval[indices]
        batch_yval  = labelval[indices]

        batch_xuval        = batch_xuval.reshape((batch_size, n_steps, n_input))
        batch_hotyval      = OneHot(batch_yval, 2)

        sess.run(optimizer_cont, feed_dict={xu: batch_xu, y: batch_hoty, kprobi: 0.5, kprobo: 0.6})
        
        if step % display_step == 0:
            acc = sess.run(accuracy, feed_dict={xu: batch_xuval, y: batch_hotyval, kprobi: 1.0, kprobo: 1.0})
	    # Calculate batch loss
            losst = sess.run(cost, feed_dict={xu: batch_xu, y: batch_hoty, kprobi: 1.0, kprobo: 1.0})
            lossv = sess.run(cost, feed_dict={xu: batch_xuval, y: batch_hotyval, kprobi: 1.0, kprobo: 1.0})
            print("Iter " + str(step*batch_size) + ", Minibatch LossT= " + \
                  "{:.6f}".format(losst) + ", Minibatch LossV= " + \
                  "{:.6f}".format(lossv) +", Training Accuracy= " + \
                  "{:.5f}".format(acc))
                  
        step += 1
    print("Optimization Finished!")
    saver.save(sess, 'anger/model/model.ckpt')






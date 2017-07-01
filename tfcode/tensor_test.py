from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import glob, os

from compact_bilinear_pooling import compact_bilinear_pooling_layer

# Network Parameters
n_input   = 256*6*6  # Dimension of emotion vector
n_steps   = 15         # Timesteps of LSTM
n_hidden  = 64       # LSTM hidden dimension
batch_size = 18
n_classes = 2         # Number of classes (real(1) or fake(0))

xu = tf.placeholder("float", [None, n_steps, n_input]) # upper face feature

# Weights
W = {'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))}
b = {'out': tf.Variable(tf.random_normal([n_classes]))}

def cbp(x1, x2, in_size, dim, size):
      v  = compact_bilinear_pooling_layer(x1, x2, in_size, dim, sum_pool=True)
      v.set_shape([size, dim])

      return v

def RNNu(xu):
    xu_ = tf.unstack(xu, n_steps, 1)

    xux = []
    for t in xrange(0, n_steps, 3):
        xmp = []
	
        xmp.append( tf.reshape(xu_[t+0], [-1, 36, 256]) )
        xmp.append( tf.reshape(xu_[t+1], [-1, 36, 256]) )
        xmp.append( tf.reshape(xu_[t+2], [-1, 36, 256]) )

        xmt = tf.transpose(xmp, perm=[1, 0, 2, 3])
        v = cbp(xmt, xmt, 256, 64, batch_size)

        xux.append( v )

    with tf.name_scope("fwu"),tf.variable_scope('fwu'):
          lstm_u_cell  = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0, activation=tf.nn.relu)
          outputs_u, states_u = rnn.static_rnn(lstm_u_cell, xux, dtype=tf.float32)

    return tf.nn.l2_normalize(outputs_u[-1], dim=1)

# FC network for classification
def DNN(x, W, b):
    out = tf.matmul(x, W) + b
    return out

# Operations
pred = RNNu(xu)
pred = tf.reshape(pred,[1, 1, batch_size, 64])
pred1 = cbp(pred, pred, 64, 512, batch_size) 
#predp1 = DNN(pred1, W['out1'], b['out1'])

# Load parameters
saver = tf.train.Saver()

with tf.Session() as sess:

    saver = tf.train.import_meta_graph('anger/model/model.ckpt.meta')
    saver.restore(sess, tf.train.latest_checkpoint('anger/model/'))
    
    filet = glob.glob('anger/train/*.npy')
    length = len(filet) 
    
    for t in xrange(0, length, 1):
        datau = np.load(filet[t])
	datau = datau.reshape((-1, n_steps, n_input))

        ts = sess.run(pred1, feed_dict={xu: datau})
        fo = 'anger/feattrain/' + os.path.os.path.basename(filet[t]) + '.tf.npy'
        print(fo)
        np.save(fo, ts)
        

    filev = glob.glob('anger/val/*.npy')
    length = len(filev)

    for t in xrange(0, length, 1):
        datau = np.load(filev[t])
        datau = datau.reshape((-1, n_steps, n_input))

        ts = sess.run(pred1, feed_dict={xu: datau})
        fo = 'anger/featval/' + os.path.os.path.basename(filev[t]) + '.tf.npy'
        print(fo)
        np.save(fo, ts)

    filete = glob.glob('anger/test/*.npy')
    length = len(filete)

    for t in xrange(0, length, 1):
        datau = np.load(filete[t])
        datau = datau.reshape((-1, n_steps, n_input))

        ts = sess.run(pred1, feed_dict={xu: datau})
        fo = 'anger/feattest/' + os.path.os.path.basename(filete[t]) + '.tf.npy'
        print(fo)
        np.save(fo, ts)


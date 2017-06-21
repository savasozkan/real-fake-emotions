from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import glob, os


# Network Parameters
n_input   = 2*96*6*3  # Dimension of emotion vector
n_steps   = 10         # Timesteps of LSTM
n_hidden  = 512       # LSTM hidden dimension
n_hidden2 = 128        # Hidden feature dimension
n_classes = 2         # Number of classes (real(1) or fake(0))

xu1 = tf.placeholder("float", [None, n_steps, n_input]) # upper face feature
xl1 = tf.placeholder("float", [None, n_steps, n_input]) # lower face feature

# Weights
W = {'out1': tf.Variable(tf.random_normal([n_hidden2, n_classes])),
     'proj1': tf.Variable(tf.random_normal([2*n_hidden, n_hidden2]))}
b = {'out1': tf.Variable(tf.random_normal([n_classes])),
     'proj1': tf.Variable(tf.random_normal([n_hidden2]))}

# LSTM network for upper and lower face features
def RNNu(xu, xl, W, b):
    xu_ = tf.unstack(xu, n_steps, 1)
    xl_ = tf.unstack(xl, n_steps, 1)

    xux = []
    xlx = []

    for t in xrange(0, n_steps, 1):
         xux.append(xu_[t])
         xlx.append(xl_[t])

    with tf.name_scope("fwu"),tf.variable_scope('fwu'):
          lstm_u_cell  = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
          outputs_u, states_u = rnn.static_rnn(lstm_u_cell, xux, dtype=tf.float32)

    with tf.name_scope("fwl"),tf.variable_scope('fwl'):
          lstm_l_cell    = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
          outputs_l, states_l = rnn.static_rnn(lstm_l_cell, xlx, dtype=tf.float32)

    xfu = outputs_u[-1]
    xfl = outputs_l[-1]
    xf_ = tf.concat([xfu, xfl], 1)
    xf  = tf.nn.relu(tf.matmul(xf_, W) + b)

    return xf

# FC network for classification
def DNN(x, W1, b1):
    out = tf.matmul(x, W1) + b1
    return out

def contractive_loss(pred1, pred2, ycont):
    margin = 4.0
    labels_t = ycont;
    labels_f = tf.subtract(1.0, ycont)
    
    eucd2 = tf.pow(tf.subtract(pred1, pred2), 2)
    eucd2 = tf.reduce_sum(eucd2, 1)
    eucd = tf.sqrt(eucd2+1e-6, name="eucd")
    C = tf.constant(margin, name="C")

    pos = tf.multiply(labels_t, eucd2, name="yi_x_eucd2")
    neg = tf.multiply(labels_f, tf.pow(tf.maximum(tf.subtract(C, eucd), 0), 2))
    losses = tf.add(pos, neg, name="losses")
    loss = tf.reduce_mean(losses, name="loss")
    
    return loss

def prediction_loss(pred, y):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))

# Operations
pred1  = RNNu(xu1, xl1, W['proj1'], b['proj1'])
predp1 = DNN(pred1, W['out1'], b['out1'])

# Load parameters
saver = tf.train.Saver()

with tf.Session() as sess:

    saver = tf.train.import_meta_graph('anger/model/model.ckpt.meta')
    saver.restore(sess, tf.train.latest_checkpoint('anger/model/'))
    
    vupper = glob.glob('anger/train/*.upper.npy')
    vlower = glob.glob('anger/train/*.lower.npy')
    
    for t in xrange(0, 80, 1):
        datau = np.load(vupper[t])
        datal = np.load(vlower[t])
	datau = datau.reshape((-1, n_steps, n_input))
	datal = datal.reshape((-1, n_steps, n_input))

        ts = sess.run(pred1, feed_dict={xu1: datau, xl1: datal})
        fo = 'anger/feattrain/' + os.path.os.path.basename(vupper[t]) + '.tf.npy'
        print(fo)
        np.save(fo, ts)
        

    vupper = glob.glob('anger/val/*.upper.npy')
    vlower = glob.glob('anger/val/*.lower.npy')

    for t in xrange(0, 10, 1):
        datau = np.load(vupper[t])
        datal = np.load(vlower[t])
        datau = datau.reshape((-1, n_steps, n_input))
        datal = datal.reshape((-1, n_steps, n_input))

        ts = sess.run(pred1, feed_dict={xu1: datau, xl1: datal})
        fo = 'anger/featval/' + os.path.os.path.basename(vupper[t]) + '.tf.npy'
        print(fo)
        np.save(fo, ts)


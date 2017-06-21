from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np

datau  = np.load('anger/data_upper.npy') # upper face data
datal  = np.load('anger/data_lower.npy') # lower face data
label  = np.load('anger/label.npy')      # labels 

# Parameters
learning_rate  = 0.001 
training_iters = 60000 
batch_size     = 64
display_step   = 10

# Network Parameters
n_input   = 2*96*6*3  # Dimension of emotion vector
n_steps   = 10         # Timesteps of LSTM
n_hidden  = 512       # LSTM hidden dimension
n_hidden2 = 128        # Hidden feature dimension
n_classes = 2         # Number of classes (real(1) or fake(0))

xu = tf.placeholder("float", [None, n_steps, n_input]) # upper face feature
xl = tf.placeholder("float", [None, n_steps, n_input]) # lower face feature
y  = tf.placeholder("float", [None, n_classes])        # labels

# Define weights
W = {'out1': tf.Variable(tf.random_normal([n_hidden2, n_classes])),
     'proj1': tf.Variable(tf.random_normal([2*n_hidden, n_hidden2]))}
b = {'out1': tf.Variable(tf.random_normal([n_classes])),
     'proj1': tf.Variable(tf.random_normal([n_hidden2]))}

def lrelu(x, leak=0.1, name='lrelu'):
    with tf.variable_scope(name):
         f1 = 0.5*(1.0 + leak)
         f2 = 0.5*(1.0 - leak)
         return f1*x + f2*abs(x)
    
def RNNu(xu, xl, W1, b1):
    xu_ = tf.unstack(xu, n_steps, 1)
    xl_ = tf.unstack(xl, n_steps, 1)

    xux = []
    xlx = []
   
    for t in xrange(0, n_steps, 1):
	 xux.append(xu_[t])
         xlx.append(xl_[t])
 
    with tf.name_scope("fwu"),tf.variable_scope('fwu'): 
          lstm_u_cell  = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
          lstm_u_drop  = rnn.DropoutWrapper(lstm_u_cell, input_keep_prob=0.5, output_keep_prob=0.5)
          outputs_u, states_u = rnn.static_rnn(lstm_u_drop, xux, dtype=tf.float32)

    with tf.name_scope("fwl"),tf.variable_scope('fwl'):
          lstm_l_cell    = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
          lstm_l_drop  = rnn.DropoutWrapper(lstm_l_cell, input_keep_prob=0.5, output_keep_prob=0.5)
          outputs_l, states_l = rnn.static_rnn(lstm_l_drop, xlx, dtype=tf.float32)
 
    xfu = outputs_u[-1] 
    xfl = outputs_l[-1] 
    xf_ = tf.concat([xfu, xfl], 1)
    xf  = tf.nn.relu(tf.matmul(xf_, W1) + b1)

    return xf 

def DNN(x, W1, b1):
    out = tf.matmul(x, W1) + b1
    return out

def contractive_loss(pred1, pred2, ycont):
    margin = 5.0
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

def exp_loss(pred1, pred2, ycont):
    eucd = tf.abs(tf.subtract(pred1, pred2))
    eucd = tf.reduce_sum(eucd, 1)
    prob = tf.exp(tf.div(-eucd, 0.01))

    losses = ycont*prob + (1.0-ycont)*tf.maximum((1-prob), 0)
    loss = tf.reduce_mean(losses, name="loss")
    
    return loss

def cosine_loss(pred1, pred2, ycont):
    nomdist = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(pred1, pred2)), 1))
    dendist = tf.add(tf.sqrt(tf.reduce_sum(tf.square(pred1), 1)), tf.sqrt(tf.reduce_sum(tf.square(pred2), 1)))
    prob    = tf.div(nomdist, dendist)

    losses = ycont*tf.square(prob) + (1.0-ycont)*tf.square(tf.maximum((1-prob), 0))
    loss = tf.reduce_mean(losses, name="loss")

    return loss

def prediction_loss(pred, y):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))

pred  = RNNu(xu, xl, W['proj1'], b['proj1'])
predp = DNN(pred, W['out1'], b['out1'])

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

    #datan, labeln = preprocess(data, label)
    while step * batch_size < training_iters:
        indices = np.random.choice(datau.shape[0], batch_size)
        batch_xu = datau[indices]
        batch_xl = datal[indices]
        batch_y  = label[indices]

        batch_xu           = batch_xu.reshape((batch_size, n_steps, n_input))
        batch_xl           = batch_xl.reshape((batch_size, n_steps, n_input))
        batch_hoty         = OneHot(batch_y, 2)

        sess.run(optimizer_cont, feed_dict={xu: batch_xu, y: batch_hoty, xl: batch_xl})
        
        if step % display_step == 0:
            acc = sess.run(accuracy, feed_dict={xu: batch_xu, y: batch_hoty, xl: batch_xl})
            # Calculate batch loss
            loss = sess.run(cost, feed_dict={xu: batch_xu, y: batch_hoty, xl: batch_xl})
            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
                  
        step += 1
    print("Optimization Finished!")
    saver.save(sess, 'anger/model/model.ckpt')






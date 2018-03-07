import tensorflow as tf
import numpy as np
import stimulus
import time
import analysis
import pickle
from parameters import *

# Ignore "use compiled version of TensorFlow" errors
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

print('Using EI Network:\t', par['EI'])
print('Synaptic configuration:\t', par['synapse_config'], "\n")

"""
Model setup and execution
"""

class Model:

    def __init__(self, input_data, target_data, mask):

        # Load the input activity, the target data, and the training mask for this batch of trials
        self.input_data = tf.unstack(input_data, axis=1)
        self.target_data = tf.unstack(target_data, axis=1)
        self.mask = tf.unstack(mask, axis=0)

        # Build the TensorFlow graph
        self.run_model()

        # Train the model
        self.optimize()


    def run_model(self):

        """
        Run the reccurent network
        History of hidden state activity stored in self.hidden_state_hist
        """
        self.rnn_cell_loop(self.input_data, self.hidden_init)

        with tf.variable_scope('output'):
            W_out = tf.get_variable('W_out', initializer = par['w_out0'], trainable=True)
            b_out = tf.get_variable('b_out', initializer = par['b_out0'], trainable=True)

        """
        Network output
        Only use excitatory projections from the RNN to the output layer
        """
        self.y_hat = [tf.matmul(tf.nn.relu(W_out),h)+b_out for h in self.hidden_state_hist]

    def rnn_cell_loop(self, x_unstacked, h):

        """
        Initialize weights and biases
        """
        with tf.variable_scope('rnn_cell'):
            W_in = tf.get_variable('W_in', initializer = par['w_in0'], trainable=True)
            W_rnn = tf.get_variable('W_rnn', initializer = par['w_rnn0'], trainable=True)
            b_rnn = tf.get_variable('b_rnn', initializer = par['b_rnn0'], trainable=True)

        self.hidden_state_hist = []

        """
        Loop through the neural inputs to the RNN, indexed in time
        """
        for rnn_input in x_unstacked:
            h = self.rnn_cell(rnn_input, h)
            self.hidden_state_hist.append(h)

    def rnn_cell(self, rnn_input, h):

        """
        Main computation of the recurrent network
        """
        with tf.variable_scope('rnn_cell', reuse=True):
            W_in = tf.get_variable('W_in')
            W_rnn = tf.get_variable('W_rnn')
            b_rnn = tf.get_variable('b_rnn')

        """
        Update the hidden state
        Only use excitatory projections from input layer to RNN
        All input and RNN activity will be non-negative
        """
        h = tf.nn.relu(tf.matmul(W_in, tf.nn.relu(rnn_input)) + tf.matmul(W_rnn, h_post) + b_rnn) \
                       + tf.random_normal([par['n_hidden'], par['batch_train_size']], 0, par['noise_rnn'], dtype=tf.float32))

        return h

    def optimize(self):

        """
        Calculate the loss functions and optimize the weights
        """
        perf_loss = tf.reduce_mean([mask*tf.reduce_mean(tf.square(y_hat-desired_output),axis=0)
                     for (y_hat, desired_output, mask) in zip(self.y_hat, self.target_data, self.mask)])

        # L2 penalty term on hidden state activity to encourage low spike rate solutions
        spike_loss = tf.reduce_mean([par['spike_cost']*tf.reduce_mean(tf.square(h), axis=0) for h in self.hidden_state_hist])

        with tf.variable_scope('rnn_cell', reuse = True):
            W_in = tf.get_variable('W_in')
            W_rnn = tf.get_variable('W_rnn')

        self.weight_loss = par['weight_cost']*(tf.reduce_sum(tf.square(W_in)) + tf.reduce_sum(tf.square(W_rnn))

        self.loss = self.perf_loss + self.spike_loss + self.weight_loss

        opt = tf.train.AdamOptimizer(learning_rate = par['learning_rate'])
        grads_and_vars = opt.compute_gradients(self.loss)

        """
        Apply any applicable weights masks to the gradient and clip
        """
        capped_gvs = []
        for grad, var in grads_and_vars:
            capped_gvs.append((tf.clip_by_norm(grad, par['clip_max_grad_val']), var))

    self.train_op = opt.apply_gradients(capped_gvs)

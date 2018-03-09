import tensorflow as tf
import numpy as np
import stimulus
import time
import pickle
from parameters import *
import matplotlib.pyplot as plt
import sys
# Ignore "use compiled version of TensorFlow" errors
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

"""
Model setup and execution
"""

class Model:

    def __init__(self, input_data, target_data, mask, train_rnn = True):

        # Load the input activity, the target data, and the training mask for this batch of trials
        self.input_data = tf.unstack(input_data, axis=1)
        self.target_data = tf.unstack(target_data, axis=1)
        self.mask = tf.unstack(mask, axis=0)
        self.train_rnn = train_rnn

        # Load the initial hidden state activity to be used at the start of each trial
        self.hidden_init = tf.constant(par['h_init'])

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
            W0 = tf.get_variable('W0', initializer = par['w_out0_0'], trainable=True)
            b0 = tf.get_variable('b0', initializer = par['b_out0_0'], trainable=True)
            W1 = tf.get_variable('W1', initializer = par['w_out1_0'], trainable=True)
            b1 = tf.get_variable('b1', initializer = par['b_out1_0'], trainable=True)
            #W2 = tf.get_variable('W2', initializer = par['w_out2_0'], trainable=True)
            #b2 = tf.get_variable('b2', initializer = par['b_out2_0'], trainable=True)

        """
        Network output
        Only use excitatory projections from the RNN to the output layer
        """
        out0 = [tf.nn.relu(tf.matmul(W0,h)+b0) for h in self.hidden_state_hist]
        #out1 = [tf.nn.relu(tf.matmul(W1,h)+b1) for h in out0]
        self.y_hat = [tf.nn.relu(tf.matmul(W1,h)+b1) for h in out0]

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
        h = tf.nn.relu(tf.matmul(W_in, rnn_input) + tf.matmul(W_rnn, h) + b_rnn \
            + tf.random_normal([par['n_recurrent'], par['batch_train_size']], 0, par['noise_rnn'], dtype=tf.float32))

        return h

    def optimize(self):

        """
        Calculate the loss functions and optimize the weights
        """
        self.perf_loss = tf.reduce_mean([mask*tf.reduce_mean(tf.square(y_hat-desired_output),axis=0)
                     for (y_hat, desired_output, mask) in zip(self.y_hat, self.target_data, self.mask)])

        # L2 penalty term on hidden state activity to encourage low spike rate solutions
        self.spike_loss = tf.reduce_mean([par['spike_cost']*tf.reduce_mean(tf.square(h), axis=0) for h in self.hidden_state_hist])

        with tf.variable_scope('rnn_cell', reuse = True):
            W_in = tf.get_variable('W_in')
            W_rnn = tf.get_variable('W_rnn')

        self.weight_loss = par['weight_cost']*(tf.reduce_mean(tf.square(W_in)) + tf.reduce_mean(tf.square(W_rnn)))

        self.loss = self.perf_loss + self.spike_loss + self.weight_loss

        opt = tf.train.AdamOptimizer(learning_rate = par['learning_rate'])
        grads_and_vars = opt.compute_gradients(self.loss)

        """
        Apply any applicable weights masks to the gradient and clip
        """
        capped_gvs = []
        for grad, var in grads_and_vars:
            if not self.train_rnn and 'rnn_cell' in var.op.name:
                grad *= 0.
            capped_gvs.append((tf.clip_by_norm(grad, par['clip_max_grad_val']), var))

        self.train_op = opt.apply_gradients(capped_gvs)

        self.reset_weights()

    def reset_weights(self):

        reset_weights = []

        for var in tf.trainable_variables():
            if not 'rnn_cell' in var.op.name:
                if 'b' in var.op.name:
                    # reset biases to 0
                    reset_weights.append(tf.assign(var, var*0.))
                elif 'W' in var.op.name:
                    # reset weights to uniform randomly distributed
                    new_weight = tf.random_uniform(var.shape, -0.05, 0.05)
                    reset_weights.append(tf.assign(var,new_weight))

        self.reset_weights = tf.group(*reset_weights)

def main(gpu_id):

    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

    """
    Reset TensorFlow before running anything
    """
    tf.reset_default_graph()

    """
    Create the stimulus class to generate trial paramaters and input activity
    """
    stim = stimulus.Stimulus()

    """
    Define all placeholder
    """
    mask = tf.placeholder(tf.float32, shape=[par['num_time_steps'], par['batch_train_size']])
    x = tf.placeholder(tf.float32, shape=[par['num_motion_dirs'], par['num_time_steps'], par['batch_train_size']])  # input data
    y = tf.placeholder(tf.float32, shape=[par['n_output'], par['num_time_steps'], par['batch_train_size']]) # target data

    with tf.Session() as sess:

        #with tf.device("/gpu:0"):
        model = Model(x, y, mask, True)
        init = tf.global_variables_initializer()
        sess.run(init)
        t_start = time.time()

        for i in range(par['num_iterations']):

            # generate batch of batch_train_size
            trial_info = stim.generate_trial(i//par['iters_per_group'])
            """
            plt.imshow(trial_info['desired_output'][:,:, 0], aspect = 'auto')
            plt.show()
            plt.imshow(trial_info['neural_input'][:,:, 0], aspect = 'auto')
            plt.show()
            """
            _, loss, perf_loss, spike_loss, weight_loss, y_hat, state_hist = \
                sess.run([model.train_op, model.loss, model.perf_loss, model.spike_loss, \
                model.weight_loss, model.y_hat, \
                model.hidden_state_hist], {x: trial_info['neural_input'], \
                y: trial_info['desired_output'], mask: trial_info['train_mask']})

            if i == par['iters_per_group']:
                sess.run(model.reset_weights)

            iteration_time = time.time() - t_start
            if (i+1)%par['iters_between_outputs']==0 or i+1==par['num_iterations']:
                print_results(i, iteration_time, perf_loss, spike_loss, weight_loss, trial_info['desired_output'])
                y1 = np.stack(y_hat, axis=1)
                """

                ind = np.argsort(np.var(np.mean(trial_info['desired_output'],axis=2), axis=1))
                plt.subplot(2,2,1)
                plt.plot(trial_info['desired_output'][ind[-1], :, 0])
                plt.plot(y1[ind[-1], :, 0],'r')
                plt.subplot(2,2,2)
                plt.plot(trial_info['desired_output'][ind[-2], :, 0])
                plt.plot(y1[ind[-2], :, 0],'r')
                plt.subplot(2,2,3)
                plt.plot(trial_info['desired_output'][ind[-3], :, 0])
                plt.plot(y1[ind[-3], :, 0],'r')
                plt.subplot(2,2,4)
                plt.plot(trial_info['desired_output'][ind[-4], :, 0])
                plt.plot(y1[ind[-4], :, 0],'r')
                plt.show()
                """

                weights = eval_weights()
                save_results = {'weights': weights, 'par': par, 'y_out': y1, 'y_target':  trial_info['desired_output'], \
                    'h': state_hist, 'input': trial_info['neural_input']}
                pickle.dump(save_results, open(par['save_dir'] + 'saved_results_wc1e6_v0.pkl', 'wb'))



def eval_weights():

    with tf.variable_scope('rnn_cell', reuse=True):
        W_in = tf.get_variable('W_in')
        W_rnn = tf.get_variable('W_rnn')
        b_rnn = tf.get_variable('b_rnn')

    with tf.variable_scope('output', reuse=True):
        W0 = tf.get_variable('W0')
        b0 = tf.get_variable('b0')
        W1 = tf.get_variable('W1')
        b1 = tf.get_variable('b1')

    weights = {
        'w_in'  : W_in.eval(),
        'w_rnn' : W_rnn.eval(),
        'w0'    : W0.eval(),
        'w1'    : W1.eval(),
        'b_rnn' : b_rnn.eval(),
        'b0'    : b0.eval(),
        'b1'    : b1.eval()
    }

    return weights

def print_results(iter_num, iteration_time, perf_loss, spike_loss, weight_loss, out):

    print('Iter {:7d}'.format(iter_num) + ' | Time {:0.2f} s'.format(iteration_time) +
      ' | Perf loss {:0.4f}'.format(perf_loss) + ' | Spike loss {:0.4f}'.format(spike_loss) +
      ' | Weight loss {:0.4f}'.format(weight_loss) + ' | Var. output {:0.4f}'.format(np.var(out)))

try:
    gpu_id = sys.argv[1]
    main(gpu_id)
except KeyboardInterrupt:
    quit('Quit by KeyboardInterrupt')

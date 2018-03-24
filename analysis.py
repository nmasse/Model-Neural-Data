import numpy as np
import stimulus
from parameters import *

def run_simulation(weights):

    update_parameters({'batch_train_size':256})
    update_parameters({'noise_in_sd':1e-9})
    update_parameters({'noise_rnn_sd':1e-9})
    update_parameters({'n_recurrent':10})
    stim  = stimulus.Stimulus()
    trial_info = stim.generate_trial(0)
    trial_type = np.zeros((par['batch_train_size']))
    for i in range(par['batch_train_size']):
        trial_type[i] = 2*(trial_info['sample'][i]//3) \
            + trial_info['test'][i]//3

    trial_length = trial_info['neural_input'].shape[1]
    x = np.split(trial_info['neural_input'],trial_length,axis=1)
    y_hat, hidden_state_hist = run_model(x, par['h_init'], weights)

    return y_hat, hidden_state_hist, trial_type


def run_model(x, hidden_init, weights):
    """
    Run the reccurent network
    History of hidden state activity stored in self.hidden_state_hist
    """
    hidden_state_hist = rnn_cell_loop(x, hidden_init, weights)

    """
    Network output
    Only use excitatory projections from the RNN to the output layer
    """
    #y0 = [np.maximum(0,np.dot(weights['w0'], h) + weights['b0']) for h in hidden_state_hist]
    #y_hat = [np.maximum(0,np.dot(weights['w1'], h) + weights['b1']) for h in y0]
    y_hat = [np.maximum(0,np.dot(weights['w0'], h) + weights['b0']) for h in hidden_state_hist]
    hidden_state_hist = np.stack(hidden_state_hist, axis=1)
    y_hat = np.stack(y_hat, axis=1)

    return y_hat, hidden_state_hist


def rnn_cell_loop(x_unstacked, h, weights):

    hidden_state_hist = []

    """
    Loop through the neural inputs to the RNN, indexed in time
    """

    for t, rnn_input in enumerate(x_unstacked):

        h = rnn_cell(np.squeeze(rnn_input), h, weights)
        hidden_state_hist.append(h)

    return hidden_state_hist

def rnn_cell(rnn_input, h, weights):

    """
    Update the hidden state
    All needed rectification has already occured
    """

    """
    h = np.maximum(0, np.dot(weights['w_in'], rnn_input) + \
        np.dot(weights['w_rnn'], h) + weights['b_rnn'] + \
        np.random.normal(0, par['noise_rnn'],size=(par['n_hidden'], par['batch_train_size'])))
    """
    h = np.maximum(0, np.dot(weights['w_in'], rnn_input) + \
        np.dot(weights['w_rnn'], h) + weights['b_rnn'] + \
        np.random.normal(0, par['noise_rnn'],size=(10, 256)))
    return h

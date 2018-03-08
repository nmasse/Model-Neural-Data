import numpy as np
import tensorflow as tf
import os
import scipy.io as sio

print("--> Loading parameters...")

global par


par = {
    # Setup parameters
    'save_dir'              : './savedir/',
    'analyze_model'         : True,


    # Network shape
    'n_recurrent'           : 5,
    'n_hidden'              : [20],

    # Timings and rates
    'dt'                    : 10,
    'learning_rate'         : 1e-3,

    # Areas to include
    'areas'                 : [1],

    # Variance values
    'clip_max_grad_val'     : 1,
    'input_mean'            : 0.0,
    'noise_in_sd'           : 0.05,
    'noise_rnn_sd'          : 0.25,

    # Tuning function data
    'num_motion_dirs'       : 6,

    # Cost parameters
    'spike_cost'            : 0.,
    'weight_cost'           : 1e-7,


    # Training specs
    'batch_train_size'      : 256,
    'num_iterations'        : 2000,
    'iters_between_outputs' : 10,

    # Task specs
    'dead_time'             : 100,
    'fix_time'              : 400,
    'sample_time'           : 660,
    'delay_time'            : 1010,
    'test_time'             : 430,


    # Save paths
    'save_fn'               : 'model_results.pkl',
}


def update_parameters(updates):
    """
    Takes a list of strings and values for updating parameters in the parameter dictionary
    Example: updates = [(key, val), (key, val)]
    """
    print('Updating parameters...')
    for key, val in updates.items():
        par[key] = val
        print('Updating ', key)

    update_trial_params()
    update_dependencies()


def update_dependencies():
    """
    Updates all parameter dependencies
    """

    data_dir = '/home/masse/'
    data = sio.loadmat(data_dir + 'spike_trains.mat')
    par['neuron_ind'] = [i for i in range(len(data['area'])) \
        if data['area'][i][0] in par['areas'] and np.mean(data['spike_train'][:,:,:,i]) < 99]
    par['n_output'] = len(par['neuron_ind'])
    par['noise_rnn'] = 1.*par['noise_rnn_sd']
    par['noise_in'] = 1.*par['noise_in_sd'] # since term will be multiplied by par['alpha_neuron']

    par['trial_length'] = par['dead_time']+par['fix_time']+par['sample_time']+par['delay_time']+par['test_time']
    # Length of each trial in time steps
    par['num_time_steps'] = par['trial_length']//par['dt']


    ####################################################################
    ### Setting up assorted intial weights, biases, and other values ###
    ####################################################################

    par['h_init'] = 0.1*np.ones((par['n_recurrent'], par['batch_train_size']), dtype=np.float32)

    par['input_to_rnn_dims'] = [par['n_recurrent'], par['num_motion_dirs']]
    par['hidden_to_hidden_dims'] = [par['n_recurrent'], par['n_recurrent']]


    # Initialize input weights
    par['w_in0'] = initialize([par['n_recurrent'], par['num_motion_dirs']])
    par['w_rnn0'] = initialize([par['n_recurrent'], par['n_recurrent']])
    par['w_out0_0'] = initialize([par['n_hidden'][0], par['n_recurrent']])
    par['w_out1_0'] = initialize([par['n_output'], par['n_hidden'][0]])

    par['b_rnn0'] = np.zeros((par['n_recurrent'], 1), dtype=np.float32)
    par['b_out0_0'] = np.zeros((par['n_hidden'][0], 1), dtype=np.float32)
    par['b_out1_0'] = np.zeros((par['n_output'], 1), dtype=np.float32)


def initialize(dims):
    #w = np.random.gamma(shape=0.25, scale=1.0, size=dims)
    w = np.random.uniform(-0.05,0.05, size=dims)
    return np.float32(w)


update_dependencies()

print("--> Parameters successfully loaded.\n")

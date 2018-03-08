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
    'n_recurrent'           : 10,
    'n_hidden'              : [25],
    'connection_prob'       : 1,         # Usually 1

    # Timings and rates
    'dt'                    : 10,
    'learning_rate'         : 1e-3,

    # Areas to include
    'areas'                 : [1,2,3],

    # Variance values
    'clip_max_grad_val'     : 1,
    'input_mean'            : 0.0,
    'noise_in_sd'           : 0.05,
    'noise_rnn_sd'          : 0.25,

    # Tuning function data
    'num_motion_dirs'       : 6,

    # Cost parameters
    'spike_cost'            : 0.,
    'wiring_cost'           : 1e-7,


    # Training specs
    'batch_train_size'      : 256,
    'num_iterations'        : 2000,
    'iters_between_outputs' : 50,

    # Task specs
    'trial_type'            : 'DMS', # allowable types: DMS, DMRS45, DMRS90, DMRS180, DMC, DMS+DMRS, ABBA, ABCA, dualDMS
    'rotation_match'        : 0,  # angular difference between matching sample and test
    'dead_time'             : 250,
    'fix_time'              : 400,
    'sample_time'           : 400,
    'delay_time'            : 800,
    'test_time'             : 400,
    'variable_delay_max'    : 600,
    'mask_duration'         : 50,  # duration of traing mask after test onset
    'catch_trial_pct'       : 0.0,
    'num_receptive_fields'  : 1,
    'num_rules'             : 1, # this will be two for the DMS+DMRS task
    'decoding_test_mode'    : False,

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

    data_dir = '/home/masse/Downloads/'
    data = sio.loadmat(data_dir + 'spike_trains.mat')
    par['neuron_ind'] = [i for in data['area'] if i in par['areas']]
    par['n_output'] = len(par['neuron_ind'])

    par['noise_rnn'] = np.sqrt(2*par['alpha_neuron'])*par['noise_rnn_sd']
    par['noise_in'] = np.sqrt(2/par['alpha_neuron'])*par['noise_in_sd'] # since term will be multiplied by par['alpha_neuron']

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


update_trial_params()
update_dependencies()

print("--> Parameters successfully loaded.\n")

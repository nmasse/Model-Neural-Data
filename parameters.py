import numpy as np
import tensorflow as tf
import os
import scipy.io as sio
import matplotlib.pyplot as plt

print("--> Loading parameters...")

global par


par = {
    # Setup parameters
    'save_dir'              : './savedir/',
    'analyze_model'         : True,


    # Network shape
    'n_recurrent'           : 10,
    'n_hidden'              : [300],

    # Timings and rates
    'dt'                    : 10,
    'learning_rate'         : 5e-3,

    # Areas to include
    'areas'                 : [2,3],

    # Variance values
    'clip_max_grad_val'     : 1,
    'input_mean'            : 0.0,
    'noise_in_sd'           : 0.05,
    'noise_rnn_sd'          : 0.1,

    # Tuning function data
    'num_motion_dirs'       : 6,

    # Cost parameters
    'spike_cost'            : 0.,
    'weight_cost'           : 1e-6,


    # Training specs
    'batch_train_size'      : 1024,
    'num_iterations'        : 8000,
    'iters_between_outputs' : 100,
    'iters_per_group'       : 4000,

    # Task specs
    'dead_time'             : 320,
    'fix_time'              : 100,
    'sample_time'           : 660,
    'delay_time'            : 1020,
    'test_time'             : 400,


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
    s = np.nanmean(np.nanmean(np.nanmean(data['spike_train'],axis=3),axis=1),axis=0)
    #plt.plot(s[200:230])
    #plt.show()
    ind = np.array([int(i) for i in range(len(data['area'])) \
        if data['area'][i][0] in par['areas'] and np.mean(data['spike_train'][:,:,:,i]) < 99])
    # neural data will be split into two equakl groups for trainig and testing the RNN
    # each will have size N
    N = len(ind)//2
    q = np.int16(np.random.permutation(N*2))
    par['neuron_ind'] = [ind[q[:N]], ind[q[N:]]]

    par['n_output'] = N
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
    #par['w_rnn0'] = 0.3*np.eye(par['n_recurrent'], dtype = np.float32)
    par['b_rnn0'] = np.zeros((par['n_recurrent'], 1), dtype=np.float32)

    par['w_out0_0'] = initialize([par['n_hidden'][0], par['n_recurrent']])
    par['b_out0_0'] = np.zeros((par['n_hidden'][0], 1), dtype=np.float32)

    """
    par['w_out1_0'] = initialize([par['n_hidden'][1], par['n_hidden'][0]])
    par['w_out2_0'] = initialize([par['n_output'], par['n_hidden'][1]])
    par['b_out1_0'] = np.zeros((par['n_hidden'][1], 1), dtype=np.float32)
    par['b_out2_0'] = np.zeros((par['n_output'], 1), dtype=np.float32)
    """
    par['w_out1_0'] = initialize([par['n_output'], par['n_hidden'][0]])
    par['b_out1_0'] = np.zeros((par['n_output'], 1), dtype=np.float32)


def initialize(dims):
    #w = np.random.gamma(shape=0.25, scale=1.0, size=dims)
    w = np.random.uniform(-0.05,0.05, size=dims)
    return np.float32(w)


update_dependencies()

print("--> Parameters successfully loaded.\n")

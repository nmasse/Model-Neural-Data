import numpy as np
import matplotlib.pyplot as plt
from parameters import *
import scipy.io as sio


class Stimulus:

    def __init__(self):

        data_dir = '/home/masse/'
        data = sio.loadmat(data_dir + 'spike_trains.mat')
        self.spike_data = np.squeeze(data['spike_train'][:,:,:, par['neuron_ind']])

    def generate_trial(self):

        trial_info = self.generate_neural_data_trial()
        return trial_info

    def generate_neural_data_trial(self):

        eodead = par['dead_time']//par['dt']
        eof = (par['dead_time']+par['fix_time'])//par['dt']
        eos = (par['dead_time']+par['fix_time']+par['sample_time'])//par['dt']
        eod = (par['dead_time']+par['fix_time']+par['sample_time']+par['delay_time'])//par['dt']
        eot = (par['dead_time']+par['fix_time']+par['sample_time']+par['delay_time']+par['test_time'])//par['dt']
        trial_info = {'desired_output'  :  np.zeros((par['n_output'], par['num_time_steps'], par['batch_train_size']),dtype=np.float32),
                      'train_mask'      :  np.ones((par['num_time_steps'], par['batch_train_size']),dtype=np.float32),
                      'sample'          :  np.zeros((par['batch_train_size']),dtype=np.int8),
                      'test'            :  np.zeros((par['batch_train_size']),dtype=np.int8),
                      'neural_input'    :  np.random.normal(par['input_mean'], par['noise_in'], size=(par['num_motion_dirs'], par['num_time_steps'], par['batch_train_size']))}


        trial_info['train_mask'][:eodead, :] = 0

        for t in range(par['batch_train_size']):

            trial_info['sample'][t] = np.random.randint(par['num_motion_dirs'])
            trial_info['test'][t] = np.random.randint(par['num_motion_dirs'])

            # SAMPLE stimulus
            trial_info['neural_input'][trial_info['sample'][t], eof:eos, t] += 1
            # TEST stimulus
            trial_info['neural_input'][trial_info['test'][t], eod:eot, t] += 1

            """
            Desired outputs
            """
            trial_info['desired_output'][:,:, t] = \
                np.transpose(self.spike_data[trial_info['sample'][t], trial_info['test'][t], :, :])

        return trial_info

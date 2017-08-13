"""
Functions used to save model data and to perform analysis
"""

import numpy as np
from parameters import *
from sklearn import svm
import time
import pickle

def analyze_model(trial_info, y_hat, h, syn_x, syn_u, model_performance, weights):

    """
    Converts neuronal and synaptic values, stored in lists, into 3D arrays
    Creating new variable since h, syn_x, and syn_u are class members of model.py,
    and will get mofiied by functions within analysis.py
    """
    syn_x_stacked = np.stack(syn_x, axis=1)
    syn_u_stacked = np.stack(syn_u, axis=1)
    h_stacked = np.stack(h, axis=1)

    """
    Calculate the neuronal and synaptic contributions towards solving the task
    """
    accuracy, accuracy_neural_shuffled, accuracy_syn_shuffled = \
        simulate_network(trial_info, h_stacked, syn_x_stacked, syn_u_stacked, weights, num_reps = 100)

    """
    Downsample neural activity in order to speed up decoding and tuning calculations
    """
    h_stacked, syn_x_stacked, syn_u_stacked, trial_time = downsample_activity(h_stacked, syn_x_stacked, syn_u_stacked, target_dt = 10)


    """
    Calculate neuronal and synaptic sample motion tuning
    """
    neuronal_pref_dir, neuronal_pev, synaptic_pref_dir, synaptic_pev = calculate_sample_tuning(h_stacked, \
        syn_x_stacked, syn_u_stacked, trial_info['sample'], trial_info['rule'], trial_info['match'], trial_time)


    """
    Decode the sample direction from neuronal activity and synaptic efficacies
    using support vector machhines
    """
    neuronal_decoding, synaptic_decoding = calculate_svms(h_stacked, syn_x_stacked, syn_u_stacked, trial_info['sample'], \
        trial_info['rule'], trial_info['match'], trial_time, num_reps = 100)


    """
    Save the results
    """
    results = {
        'neuronal_decoding': neuronal_decoding,
        'synaptic_decoding': synaptic_decoding,
        'neuronal_pref_dir': neuronal_pref_dir,
        'neuronal_pev': neuronal_pev,
        'synaptic_pref_dir': synaptic_pref_dir,
        'synaptic_pev': synaptic_pev,
        'accuracy': accuracy,
        'accuracy_neural_shuffled': accuracy_neural_shuffled,
        'accuracy_syn_shuffled': accuracy_syn_shuffled,
        'model_performance': model_performance,
        'parameters': par,
        'weights': weights,
        'trial_time': trial_time}

    save_fn = par['save_dir'] + par['save_fn']
    pickle.dump(results, open(save_fn, 'wb') )
    print('Analysis results saved in ', save_fn)

def calculate_svms(h, syn_x, syn_u, sample, rule, match, trial_time, num_reps = 20):

    """
    Calculates neuronal and synaptic decoding accuracies uisng support vector machines
    sample is the index of the sample motion direction for each trial_length
    rule is the rule index for each trial_length
    """

    lin_clf = svm.SVC(C=1, kernel='linear', decision_function_shape='ovr', shrinking=False, tol=1e-4)

    num_time_steps = len(trial_time)
    neuronal_decoding = np.zeros((par['num_rules'], num_time_steps, num_reps))
    synaptic_decoding = np.zeros((par['num_rules'], num_time_steps, num_reps))

    """
    The synaptic efficacy is the product of syn_x and syn_u, will decode sample
    direction from this value
    """
    syn_efficacy = syn_x*syn_u

    if par['trial_type'] == 'DMC':
        """
        Will also calculate the category decoding accuracies, assuming the first half of
        the sample direction belong to category 1, and the second half belong to category 2
        """
        num_motion_dirs = len(np.unique(sample))
        sample = np.floor(sample/(num_motion_dirs/2)*np.ones_like(sample))

    elif par['trial_type'] == 'ABBA' or par['trial_type'] == 'ABCA':
        """
        For ABBA/ABCA trials, will only analyze trials for which the first n-1
        test stimuli, out of n, are non-matches
        """
        ind = np.where(np.sum(match[:,:-1],axis=1)==0)[0]
        sample = sample[ind]
        rule = rule[ind]
        h = h[:,:,ind]
        syn_efficacy = syn_efficacy[:,:,ind]

    # number of unique samples
    N = len(np.unique(sample))
    neuronal_decoding, synaptic_decoding = svm_wraper(lin_clf, h, syn_efficacy, sample, rule, num_reps, N, trial_time)

    return neuronal_decoding, synaptic_decoding

def svm_wraper(lin_clf, h, syn_eff, conds, rule, num_reps, num_conds, trial_time):

    """
    Wraper function used to decode sample direction from hidden activity (h)
    and synaptic efficacies (syn_eff)
    """
    train_pct = 0.75
    trials_per_cond = 25
    _, num_time_steps, num_trials = h.shape
    score_h = np.zeros((par['num_rules'], num_reps, num_time_steps))
    score_syn_eff = np.zeros((par['num_rules'], num_reps, num_time_steps))

    for r in range(par['num_rules']):
        ind_rule = np.where(rule==r)[0]
        for rep in range(num_reps):
            q = np.random.permutation(len(ind_rule))
            i = int(np.round(len(ind_rule)*train_pct))
            train_ind = ind_rule[q[:i]]
            test_ind = ind_rule[q[i:]]
            equal_train_ind = np.zeros((num_conds*trials_per_cond), dtype = np.uint16)
            equal_test_ind = np.zeros((num_conds*trials_per_cond), dtype = np.uint16)

            for c in range(num_conds):
                u = range(c*trials_per_cond, (c+1)*trials_per_cond)
                # training indices for current condition number
                ind = np.where(conds[train_ind] == c)[0]
                q = np.random.randint(len(ind), size = trials_per_cond)
                equal_train_ind[u] =  train_ind[ind[q]]
                # testing indices for current condition number
                ind = np.where(conds[test_ind] == c)[0]
                q = np.random.randint(len(ind), size = trials_per_cond)
                equal_test_ind[u] =  test_ind[ind[q]]

            for t in range(num_time_steps):
                if trial_time[t] <= par['dead_time']:
                    # no need to analyze activity during dead time
                    continue

                score_h[r,rep,t] = calc_svm(lin_clf, h[:,t,:].T, conds, equal_train_ind, equal_test_ind)
                score_syn_eff[r,rep,t] = calc_svm(lin_clf, syn_eff[:,t,:].T, conds, equal_train_ind, equal_test_ind)

    return score_h, score_syn_eff



def calc_svm(lin_clf, y, conds, train_ind, test_ind):

    # normalize values between 0 and 1
    # only include feature (i.e neurons or synapses) whose min and max values differ

    """
    feature_ind = []
    for i in range(y.shape[1]):
        m1 = y[:,i].min()
        m2 = y[:,i].max()
        y[:,i] -= m1
        if m2>m1:
            y[:,i] /=(m2-m1)
            feature_ind.append(i)

    y = y[:, feature_ind]
    """

    lin_clf.fit(y[train_ind,:], conds[train_ind])
    dec = lin_clf.predict(y[test_ind,:])
    score = 0
    for i in range(len(test_ind)):
        if conds[test_ind[i]]==dec[i]:
            score += 1/len(test_ind)

    return score

def simulate_network(trial_info, h, syn_x, syn_u, weights, num_reps = 20):

    """
    Simulation will start from the start of the test period until the end of trial
    """
    if par['trial_type'] == 'dualDMS':
        test_onset = (par['dead_time']+par['fix_time']+par['sample_time']+2*par['delay_time']+par['test_time'])//par['dt']
    else:
        test_onset = (par['dead_time']+par['fix_time']+par['sample_time']+par['delay_time'])//par['dt']

    accuracy = np.zeros((par['num_rules'], num_reps))
    accuracy_neural_shuffled = np.zeros((par['num_rules'], num_reps))
    accuracy_syn_shuffled = np.zeros((par['num_rules'], num_reps))

    _, trial_length, batch_train_size = h.shape
    test_length = trial_length - test_onset

    for r in range(par['num_rules']):
        # For ABBA/ABCA trials, will only analyze trials for which the first n-1
        # test stimuli, out of n, are non-matches
        if par['trial_type'] == 'ABBA' or par['trial_type'] == 'ABCA':
            trial_ind = np.where((np.sum(match[:,:-1],axis=1)==0)*(trial_info['rule']==r))[0]
        else:
            trial_ind = np.where(trial_info['rule']==r)[0]
        train_mask = trial_info['train_mask'][test_onset:,trial_ind]
        x = np.split(trial_info['neural_input'][:,test_onset:,trial_ind],test_length,axis=1)
        y = trial_info['desired_output'][:,test_onset:,trial_ind]

        for n in range(num_reps):

            """
            Calculating behavioral accuracy without shuffling
            """
            hidden_init = h[:,test_onset-1,trial_ind]
            syn_x_init = syn_x[:,test_onset-1,trial_ind]
            syn_u_init = syn_u[:,test_onset-1,trial_ind]
            y_hat = run_model(x, y, hidden_init, syn_x_init, syn_u_init, weights)
            accuracy[r,n] = get_perf(y, y_hat, train_mask)

            """
            Keep the synaptic values fixed, permute the neural activity
            """
            ind_shuffle = np.random.permutation(len(trial_ind))

            hidden_init = hidden_init[:,ind_shuffle]
            y_hat = run_model(x, y, hidden_init, syn_x_init, syn_u_init, weights)
            accuracy_neural_shuffled[r,n] = get_perf(y, y_hat, train_mask)

            """
            Keep the hidden values fixed, permute synaptic values
            """
            hidden_init = h[:,test_onset-1,trial_ind]
            syn_x_init = syn_x_init[:,ind_shuffle]
            syn_u_init = syn_u_init[:,ind_shuffle]
            y_hat = run_model(x, y, hidden_init, syn_x_init, syn_u_init, weights)
            accuracy_syn_shuffled[r,n] = get_perf(y, y_hat, train_mask)

    return accuracy, accuracy_neural_shuffled, accuracy_syn_shuffled

def calculate_sample_tuning(h, syn_x, syn_u, sample, rule, match, trial_time):

    """
    Calculates neuronal and synaptic sample motion direction tuning
    """
    num_time_steps = len(trial_time)
    neuronal_pref_dir = np.zeros((par['n_hidden'],  par['num_rules'], num_time_steps))
    synaptic_pref_dir = np.zeros((par['n_hidden'],  par['num_rules'], num_time_steps))
    neuronal_pev = np.zeros((par['n_hidden'],  par['num_rules'], num_time_steps))
    synaptic_pev = np.zeros((par['n_hidden'],  par['num_rules'], num_time_steps))

    """
    The synaptic efficacy is the product of syn_x and syn_u, will decode sample
    direction from this value
    """
    syn_efficacy = syn_x*syn_u

    if par['trial_type'] == 'ABBA' or par['trial_type'] == 'ABCA':
        """
        For ABBA/ABCA trials, will only analyze trials for which the first n-1
        test stimuli, out of n, are non-matches
        """
        ind = np.where(np.sum(match[:,:-1],axis=1)==0)[0]
        sample = sample[ind]
        rule = rule[ind]
        h = h[:,:,ind]
        syn_efficacy = syn_efficacy[:,:,ind]

    # number of unique samples
    N = len(np.unique(sample))

    sample_dir = np.ones((len(sample), 3))
    sample_dir[:,1] = np.cos(2*np.pi*sample/N)
    sample_dir[:,2] = np.sin(2*np.pi*sample/N)

    for r in range(par['num_rules']):
        ind = np.where((rule==r))[0]
        for n in range(par['n_hidden']):
            for t in range(num_time_steps):
                if trial_time[t] <= par['dead_time']:
                    # no need to analyze activity during dead time
                    continue

                # Neuronal sample tuning
                weights = np.linalg.lstsq(sample_dir[ind,:], h[n,t,ind])
                weights = np.reshape(weights[0],(3,1))
                pred_err = h[n,t,ind] - np.dot(sample_dir[ind,:], weights).T
                mse = np.mean(pred_err**2)
                response_var = np.var(h[n,t,ind])
                neuronal_pev[n,r,t] = 1 - mse/(response_var+1e-9)
                neuronal_pref_dir[n,r,t] = np.arctan2(weights[2,0],weights[1,0])

                # Synaptic sample tuning
                weights = np.linalg.lstsq(sample_dir[ind,:], syn_efficacy[n,t,ind])
                weights = np.reshape(weights[0],(3,1))
                pred_err = syn_efficacy[n,t,ind] - np.dot(sample_dir[ind,:], weights).T
                mse = np.mean(pred_err**2)
                response_var = np.var(syn_efficacy[n,t,ind])
                synaptic_pev[n,r,t] = 1 - mse/(response_var+1e-9)
                synaptic_pref_dir[n,r,t] = np.arctan2(weights[2,0],weights[1,0])

    return neuronal_pref_dir, neuronal_pev, synaptic_pref_dir, synaptic_pev


def run_model(x, y, hidden_init, syn_x_init, syn_u_init, weights):

    """
    Run the reccurent network
    History of hidden state activity stored in self.hidden_state_hist
    """
    hidden_state_hist = rnn_cell_loop(x, hidden_init, syn_x_init, syn_u_init, weights)

    """
    Network output
    Only use excitatory projections from the RNN to the output layer
    """
    y_hat = [np.dot(np.maximum(0,weights['w_out']), h) + weights['b_out'] for h in hidden_state_hist]

    return y_hat


def rnn_cell_loop(x_unstacked, h, syn_x, syn_u, weights):

    hidden_state_hist = []

    """
    Loop through the neural inputs to the RNN, indexed in time
    """
    for rnn_input in x_unstacked:
        h, syn_x, syn_u = rnn_cell(np.squeeze(rnn_input), h, syn_x, syn_u, weights)
        hidden_state_hist.append(h)

    return hidden_state_hist

def rnn_cell(rnn_input, h, syn_x, syn_u, weights):

    if par['EI']:
        # ensure excitatory neurons only have postive outgoing weights,
        # and inhibitory neurons have negative outgoing weights
        W_rnn_effective = np.dot(np.maximum(0,weights['w_rnn']), par['EI_matrix'])
    else:
        W_rnn_effective = weights['w_rnn']


    """
    Update the synaptic plasticity paramaters
    """
    if par['synapse_config'] == 'std_stf':
        # implement both synaptic short term facilitation and depression
        syn_x += par['alpha_std']*(1-syn_x) - par['dt_sec']*syn_u*syn_x*h
        syn_u += par['alpha_stf']*(par['U']-syn_u) + par['dt_sec']*par['U']*(1-syn_u)*h
        syn_x = np.minimum(1, np.maximum(0, syn_x))
        syn_u = np.minimum(1, np.maximum(0, syn_u))
        h_post = syn_u*syn_x*h

    elif par['synapse_config'] == 'std':
        # implement synaptic short term derpression, but no facilitation
        # we assume that syn_u remains constant at 1
        syn_x += par['alpha_std']*(1-syn_x) - par['dt_sec']*syn_x*h
        syn_x = np.minimum(1, np.maximum(0, syn_x))
        h_post = syn_x*h

    elif par['synapse_config'] == 'stf':
        # implement synaptic short term facilitation, but no depression
        # we assume that syn_x remains constant at 1
        syn_u += par['alpha_stf']*(par['U']-syn_u) + par['dt_sec']*par['U']*(1-syn_u)*h
        syn_u = np.minimum(1, np.maximum(0, syn_u))
        h_post = syn_u*h

    else:
        # no synaptic plasticity
        h_post = h

    """
    Update the hidden state
    All needed rectification has already occured
    """

    h = np.maximum(0, h*(1-par['alpha_neuron'])
                   + par['alpha_neuron']*(np.dot(np.maximum(0,weights['w_in']), np.maximum(0, rnn_input))
                   + np.dot(W_rnn_effective, h_post) + weights['b_rnn'])
                   + np.random.normal(0, par['noise_rnn'],size=(par['n_hidden'], h.shape[1])))

    return h, syn_x, syn_u

def downsample_activity(h, syn_x, syn_u, target_dt = 10):

    # downsample activity
    df = target_dt//par['dt']
    if df > 1:
        rng = range(0,h.shape[1], df)
        h = h[:, rng, :]
        syn_x = syn_x[:, rng, :]
        syn_u = syn_u[:, rng, :]
    elif df < 1:
        print('Cannot sample at higher temproal resolution!!!')
        quit()

    trial_time = np.arange(0,h.shape[1]*target_dt, target_dt)

    return h, syn_x, syn_u, trial_time


def get_perf(y, y_hat, mask):

    """
    Calculate task accuracy by comparing the actual network output to the desired output
    only examine time points when test stimulus is on
    in another words, when y[0,:,:] is not 0
    """

    y_hat = np.stack(y_hat, axis=1)
    mask *= y[0,:,:]==0
    y = np.argmax(y, axis = 0)
    y_hat = np.argmax(y_hat, axis = 0)

    return np.sum(np.float32(y == y_hat)*np.squeeze(mask))/np.sum(mask)

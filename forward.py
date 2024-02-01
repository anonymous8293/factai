
import numpy as np
import math
from scipy.stats import multivariate_normal
import torch as th




def get_gauss_params(hmm, hidden_state):
    '''Calculates the corresponding mean and covariance matrix'''
    mean, covariance = hmm.init_dist()
    return mean[hidden_state], covariance[hidden_state]

def E_probs(hmm, hidden_state, x):
    '''Generates the emission probability (value of the Gaussian PDF) of x given the hidden state.
        Note: the hidden state determines both the multivariate Gaussian distribution and the salient feature.
        This function calculates f(x) where f is the marginal (Gaussian) PDF of the salient feature. 
    '''
    mean, covariance = get_gauss_params(hmm, hidden_state) 
    f = multivariate_normal.pdf(x, mean, covariance)   

    return f

     

def T_probs(prev_state, new_state, time):
    '''Defines the transition probabilities of the Hidden Markov Chain'''
    p = 0.95-(float(time)/500.0) if prev_state == 1 else 0.05 + (float(time)/500.0)
    return p if new_state == 1 else 1-p



def forward(hmm, observations):
    '''Applies forward algorithm for calculating marginal probability of observations from our HMM.
       observations: datapoint for which we calculate the probability, tensor with shape = (200,3)
    '''

    M = observations.shape[0]   # length of HMM
    N = 2   # number of diff hidden states (this is not equal to n_state)
    
    alpha = np.zeros((M, N)) # this table will be filled by the forward algorithm 

    
    # Initialization
    '''
    Since there are no previous states, the probability of getting vector x at time 0 from state s is given as product of:
    1. Initial Probability of being in state s (this is 0.5)
    2. Emmision Probability of getting x from state s
    '''
    for s in range(N):
        alpha[0, s] = 0.5 * E_probs(hmm, s, observations[0])  
    
    # Induction, forward algorithm
    '''
    if we know the previous state i,then the probability of being in state j at time t+1 is given as product of:
    1. Probability of being in state i at time t
    2. Transition probability of going from state i to state j
    3. Emmision Probability of symbol O(t+1) being in state j
    '''
    for t in range(1, M):
        for s in range(N):
            for i in range(N):
                alpha[t, s] += alpha[t-1, i] * T_probs(i, s, t) * E_probs(hmm, s, observations[t])
    
    return alpha, np.sum(alpha[M-1,:])

"""
Template for implementing QLearner  (c) 2015 Tucker Balch
"""

import numpy as np
import pandas as pd
import random as rand

class QLearner(object):

    def __init__(self, \
        num_states=100, \
        num_actions = 4, \
        alpha = 0.2, \
        gamma = 1.0, \
        rar = 0.5, \
        radr = 0.99, \
        trade_penalty = 7, \
        dyna = 200, \
        verbose = False):

        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.orginalrar = rar
        self.rar = rar
        self.radr = radr
        self.trade_penalty = trade_penalty
        self.dyna = dyna
        self.verbose = verbose
        self.s = 0
        self.a = 0
        self.qvalues_array = 2*np.random.random((num_states, num_actions)) - 1
        # self.qvalues_array = np.zeros((num_states, num_actions))
        self.exp_tuples = pd.DataFrame(columns=['CurrentState','Action','NextState','Reward'])

    def querysetstate(self, s):
        """
        @summary: Update the state without updating the Q-table
        @param s: The new state
        @returns: The selected action
        """
        self.s = s
        a_prime = self.getnextaction(s)
        # if self.verbose: print "s =", s,"a =",a_prime
        self.a = a_prime
        return a_prime

    def query(self,s_prime,r):
        """
        @summary: Update the Q table and return an action
        @param s_prime: The new state
        @param r: The ne state
        @returns: The selected action
        """
        a_prime = self.getnextaction(s_prime)
        self.rar = self.rar*self.radr
        if self.verbose: print self.rar

        # Apply penalty to take trade to simulate brokerage
        if self.a != 0:
            r = r - self.trade_penalty

        qvalue_prime = self.qvalues_array[s_prime, a_prime]
        qvalue = (1-self.alpha)*self.qvalues_array[self.s, self.a] + self.alpha*(r + self.gamma*qvalue_prime)
        self.qvalues_array[self.s, self.a] = qvalue

        # Here is the implementation of Dyna
        # I'm recording the experience tuples performed by the agent in real. It contains repeated experiences as well.
        # I take out random samples directly, as the repeated exp will have higher probablity of getting chosen, so no need to calculate probability again.
        # The implementation is completely vectorized, but still it takes 13-15 sec on my machine.
        # Though, this algorithm works wonders and value converges within 5 iterations, so 50 iterations are not at all required.

        if self.dyna > 0:
            self.exp_tuples = self.exp_tuples.append({'CurrentState':self.s, 'Action':self.a,
                                                      'NextState': s_prime, 'Reward':r}, ignore_index=True)
            # print len(self.exp_tuples)
            random_index = np.random.randint(0,self.exp_tuples.shape[0], self.dyna)
            random_exp = self.exp_tuples.ix[random_index]
            dyna_s = list(random_exp['CurrentState'].values.astype(np.int64))
            dyna_a = list(random_exp['Action'].values.astype(np.int64))
            dyna_s_prime = np.array(random_exp['NextState'], dtype= int)
            dyna_r = random_exp['Reward'].values

            dyna_qvalue_prime = self.qvalues_array[dyna_s_prime].max(axis=1)
            dyna_qvalue = (1-self.alpha)*self.qvalues_array[dyna_s,dyna_a] + self.alpha*(dyna_r + self.gamma*dyna_qvalue_prime)
            self.qvalues_array[dyna_s,dyna_a] = dyna_qvalue


        # if self.verbose: print "s =", s_prime,"a =",a_prime,"r =",r
        self.s = s_prime
        self.a = a_prime

        return a_prime

    def getnextaction(self, s_prime):
        magic_number = np.random.random()
        if magic_number < self.rar:
            a_prime  = np.random.randint(0, self.num_actions)
        else:
            a_prime = self.qvalues_array[s_prime].argmax()
        return a_prime

if __name__=="__main__":
    print "Remember Q from Star Trek? Well, this isn't him"
